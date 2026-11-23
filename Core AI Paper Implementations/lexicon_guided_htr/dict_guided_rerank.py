"""Part A — Dictionary-Guided re-ranking of an existing transcription.

Faithful adaptation of *Dictionary-Guided Scene Text Recognition* (Nguyen
et al., CVPR 2021) to the Pernegg parish registers, using the Reichstelefonbuch
1942 as the per-region dictionary.

WHAT THE PAPER DOES
-------------------
The recognizer emits top-k candidate strings with per-character confidences;
a re-ranker then picks the dictionary word that minimises a *confidence-
weighted edit distance* — positions the recognizer was unsure about are cheap
to edit toward a dictionary match, confident positions are expensive. Output
is the best (beam, dictionary-entry) pair, with an OOV fallback to the raw
beam when no dictionary word scores well enough.

HOW THIS PORT DIFFERS (and why)
-------------------------------
The input here is a *finished* transcription (the Pernegg xlsx), not a live
decoder, so we have neither top-k beams nor per-character logits. Two
substitutions keep the method faithful:

  1. Confidence-weighted edit distance  →  *confusion-weighted* edit distance.
     Absent logits, substitution cost is lowered for character pairs that are
     genuinely confusable in Kurrent / Latin cursive and Fraktur print
     (n/u, m/rn, P/S, ſ/s, …). This is the same idea — make visually-ambiguous
     positions cheap to correct — driven by a script prior instead of logits.
     If live logprobs ever become available, pass `char_conf=` to
     `weighted_edit_distance` and the original paper formulation is recovered
     exactly (see `confidence_from_logprobs`).

  2. The dictionary prior is REGION-FREQUENCY weighted (user requirement):
     a surname is trusted in proportion to how often the Reichstelefonbuch
     attests it *in this region*. This both implements "give more weight to the
     RTB for names that recur in the region" and neutralises the noise caveat —
     a hapax RTB entry carries almost no prior, a 700× surname carries a lot.

SCORING
-------
    score(tok, cand) = sim(tok, cand)                       # 0..100, weighted-edit similarity
                     + W_RTB  * hop_bonus(cand)             # 1-hop town >> 2-hop state >> national
                     + W_FREQ * log1p(region_count(cand))   # RTB regional frequency = trust

A replacement is only made when the best candidate clears SIM_FLOOR (visual
plausibility — protects OOV names), beats the runner-up by MARGIN, and the
token is not already a valid lexicon word. SIM_FLOOR is the τ hyperparameter
the report says to sweep; `eval_oov` sweeps it.

Usage
-----
    # correct one page, write a copy + replacement log into the package folder
    venv/bin/python dict_guided_rerank.py correct \
        --xlsx "../../Pernegg Parish yearbook xlsx/oesterreich_graz-seckau_Pernegg_8758/Seite 11-12.xlsx" \
        --parish "Pernegg (Steiermark)"

    # OOV-segmented evaluation against the manual ground truth, sweeping tau
    venv/bin/python dict_guided_rerank.py eval \
        --gt pernegg_run/gt_seite_11-12_entities.txt \
        --xlsx "../../Pernegg Parish yearbook xlsx/oesterreich_graz-seckau_Pernegg_8758/Seite 11-12.xlsx" \
        --parish "Pernegg (Steiermark)"
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import editdistance
import pandas as pd
from rapidfuzz import fuzz, process

import config
from kg_retrieval import LexiconKG, is_noise, same_state, _norm_state

# ── output is confined to this package folder ────────────────────────────────
OUT_DIR = config.PKG_DIR / "dict_guided_run"

# ── re-ranking hyperparameters ───────────────────────────────────────────────
SIM_FLOOR = 72.0        # τ: min visual similarity to even consider a swap
MARGIN = 4.0            # best total score must beat runner-up by this much
W_RTB = 6.0             # weight on geographic attestation (hop bonus)
W_FREQ = 4.0            # weight on RTB regional frequency  ← "more weight to RTB"
HOP_BONUS = {"town": 2.0, "state": 1.0, "national": 0.0}
CONFUSION_COST = 0.35   # substitution cost for a known script-confusable pair
MAX_LEN_DELTA = 0.45    # reject candidates whose length differs too much

# German capitalised words that are not entities (never "corrected")
SKIP = {
    "Jahre", "Lebensjahre", "Krankheit", "Todesart", "Religion", "Geschlecht",
    "Namen", "Haus", "Monat", "Seite", "Uhr", "Nachts", "Abends", "Mittags",
    "Vormittag", "Nachmittag", "Ort", "Nummer", "Versehen", "Gestorbenen",
}
WORD_RE = re.compile(r"[A-ZÄÖÜ][a-zäöüß\-]{3,}")

# Character pairs that are genuinely confusable in Kurrent / Latin cursive /
# Fraktur. Lowered substitution cost — the confusion-matrix stand-in for the
# paper's per-character confidence. Pairs are unordered.
_CONFUSABLE = [
    ("n", "u"), ("m", "n"), ("c", "e"), ("o", "a"), ("i", "l"), ("i", "j"),
    ("t", "l"), ("h", "b"), ("k", "h"), ("v", "r"), ("w", "m"), ("f", "s"),
    ("s", "z"), ("g", "q"), ("u", "v"), ("e", "o"), ("r", "x"), ("a", "u"),
    # capitals seen swapped in this corpus (Pommer/Sommer, Lein/Stein …)
    ("P", "S"), ("B", "P"), ("F", "J"), ("G", "C"), ("S", "L"), ("K", "R"),
    ("H", "K"), ("T", "F"), ("ß", "s"), ("ſ", "s"),
]
_CONF = set()
for _a, _b in _CONFUSABLE:
    _CONF.add((_a, _b))
    _CONF.add((_b, _a))


# ── confidence-weighted (confusion-weighted) edit distance ───────────────────
def _sub_cost(a: str, b: str, conf: float | None) -> float:
    """Cost of substituting source char `a` with target char `b`.

    conf is the recognizer's confidence in `a` (0..1) when available: a low
    confidence makes the edit cheap (paper formulation). When conf is None we
    fall back to the script confusion prior.
    """
    if a == b:
        return 0.0
    if conf is not None:
        return 1.0 - conf            # paper-faithful path
    if (a, b) in _CONF:
        return CONFUSION_COST        # confusion-matrix stand-in
    return 1.0


def weighted_edit_distance(candidate: str, target: str,
                           char_conf: list[float] | None = None) -> float:
    """Levenshtein with a position-aware substitution cost.

    `char_conf[i]` (optional) is the recognizer's confidence in candidate[i];
    if given, this is exactly the CVPR paper's weighted edit distance. If None,
    substitution cost comes from the Kurrent/Fraktur confusion set.
    """
    n, m = len(candidate), len(target)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ci = char_conf[i - 1] if char_conf is not None and i - 1 < len(char_conf) else None
        for j in range(1, m + 1):
            sub = dp[i - 1][j - 1] + _sub_cost(candidate[i - 1], target[j - 1], ci)
            dp[i][j] = min(sub, dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0)
    return dp[n][m]


def similarity(token: str, cand: str, char_conf: list[float] | None = None) -> float:
    """0..100 visual similarity from the weighted edit distance."""
    if not token or not cand:
        return 0.0
    wed = weighted_edit_distance(token, cand, char_conf)
    return 100.0 * (1.0 - wed / max(len(token), len(cand)))


def confidence_from_logprobs(token_logprobs: list[tuple[str, float]]) -> tuple[str, list[float]]:
    """Bridge for the live-decoder case (LIMITATION of the xlsx path).

    A BPE token covers several characters but carries one logprob. We spread
    the token's probability uniformly over its characters — the documented
    approximation. Returns (string, per_char_confidence) ready for
    `weighted_edit_distance(..., char_conf=...)`. Unused by the xlsx pipeline;
    provided so live Qwen3-VL beams plug in without changing the re-ranker.
    """
    chars, confs = [], []
    for piece, logp in token_logprobs:
        p = math.exp(logp)
        for ch in piece:
            chars.append(ch)
            confs.append(p)
    return "".join(chars), confs


# ── the per-region dictionary ────────────────────────────────────────────────
@dataclass
class TownDictionary:
    """RTB vocabulary scoped to one parish, with the counts that drive trust."""
    parish: str
    town: str | None
    state: str | None
    town_surnames: set[str]                       # 1-hop (membership)
    region_count: dict[str, int]                  # 2-hop surname -> regional count
    given_names: dict[str, int]
    occupations: dict[str, int]
    places: dict[str, int]
    national_surnames: dict[str, int]
    _kg: LexiconKG = field(repr=False, default=None)

    def hop_bonus(self, cand: str) -> float:
        if cand in self.town_surnames:
            return HOP_BONUS["town"]
        if cand in self.region_count:
            return HOP_BONUS["state"]
        return HOP_BONUS["national"]

    def freq(self, cand: str) -> int:
        """Regional count if attested in-region, else national count (backstop).
        Region attestation is what the user wants weighted; the national count
        only ranks among nationally-known names that aren't region-local."""
        if cand in self.region_count:
            return self.region_count[cand]
        return self.national_surnames.get(cand, 0)

    def pool_for(self, field_type: str) -> list[str]:
        if field_type == "place":
            return list(self.places)
        if field_type == "occupation":
            return list(self.occupations)
        # name (default): surnames (region + national) + given names
        seen, out = set(), []
        for src in (self.town_surnames, self.region_count,
                    self.national_surnames, self.given_names):
            for w in src:
                if w not in seen and not is_noise(w):
                    seen.add(w)
                    out.append(w)
        return out

    def is_valid(self, word: str) -> bool:
        lex = self._kg.lex
        return (word in self.region_count or word in self.national_surnames
                or word in lex["given_names"] or word in lex["towns"]
                or word in lex["occupations"] or word in self.town_surnames)


def load_town_dictionary(parish: str = "Pernegg (Steiermark)") -> TownDictionary:
    """Build the region-scoped RTB dictionary from the prebuilt artifacts.

    1-hop = surnames attested in the parish heading itself (tiny for Pernegg);
    2-hop = surnames aggregated over the whole state, *with counts* — this is
    the 'appears often in the region' frequency the re-ranker trusts.
    """
    kg = LexiconKG.load()
    town = kg.match_town(parish)
    state = kg.town_state.get(town) if town else None

    # 1-hop: merge heading case/diacritic variants ("Pernegg (Steiermark)" ==
    # "Pernegg (steiermark)") into one town surname set
    town_surnames: set[str] = set()
    if town:
        key = _norm_state(town)
        for heading, surnames in kg.kg_raw.items():
            if _norm_state(heading) == key:
                town_surnames.update(s for s in surnames if not is_noise(s))

    # 2-hop: state-aggregated surname counts (regional frequency)
    region_count: dict[str, int] = {}
    if state:
        for heading, surnames in kg.kg_raw.items():
            st = kg.town_state.get(heading)
            if st and same_state(st, state):
                for s, c in surnames.items():
                    if not is_noise(s):
                        region_count[s] = region_count.get(s, 0) + c

    places = {t: sum(kg.kg_raw.get(t, {}).values())
              for t, st in kg.town_state.items()
              if state and same_state(st, state)}

    return TownDictionary(
        parish=parish, town=town, state=state,
        town_surnames=town_surnames,
        region_count=region_count,
        given_names={g: c for g, c in kg.lex["given_names"].items() if not is_noise(g)},
        occupations={o: c for o, c in kg.lex["occupations"].items() if not is_noise(o)},
        places=places or {t: 1 for t in kg.lex["towns"]},
        national_surnames={s: c for s, c in kg.lex["surnames"].items() if not is_noise(s)},
        _kg=kg,
    )


# ── the re-ranker ────────────────────────────────────────────────────────────
@dataclass
class Decision:
    token: str
    chosen: str
    score: float
    runner_up: float
    sim: float
    replaced: bool
    oov: bool           # no candidate cleared SIM_FLOOR -> kept original (OOV)


def rerank_token(token: str, dic: TownDictionary, field_type: str = "name",
                 char_conf: list[float] | None = None,
                 sim_floor: float = SIM_FLOOR) -> Decision:
    """Pick the best RTB dictionary entry for a (possibly garbled) token, or
    keep the token unchanged if it is already valid or genuinely OOV."""
    if token in SKIP or is_noise(token) or dic.is_valid(token):
        return Decision(token, token, 0.0, 0.0, 100.0, False, False)

    pool = dic.pool_for(field_type)
    # cheap visual prefilter, then exact weighted-edit scoring on survivors
    prelim = process.extract(token, pool, scorer=fuzz.ratio,
                             score_cutoff=sim_floor - 8, limit=25)
    scored: list[tuple[str, float, float]] = []
    for cand, _, _ in prelim:
        if abs(len(cand) - len(token)) > max(2, MAX_LEN_DELTA * len(token)):
            continue
        sim = similarity(token, cand, char_conf)
        if sim < sim_floor:
            continue
        total = sim + W_RTB * dic.hop_bonus(cand) + W_FREQ * math.log1p(dic.freq(cand))
        scored.append((cand, total, sim))

    if not scored:
        # OOV fallback — no visually-plausible dictionary word, keep the pixels
        return Decision(token, token, 0.0, 0.0, 0.0, False, True)

    scored.sort(key=lambda t: -t[1])
    best, best_total, best_sim = scored[0]
    runner = scored[1][1] if len(scored) > 1 else 0.0
    if best == token or best_total - runner < MARGIN:
        return Decision(token, token, best_total, runner, best_sim, False, False)
    return Decision(token, best, best_total, runner, best_sim, True, False)


# ── field-type inference from column headers ─────────────────────────────────
NAME_HDR = re.compile(r"gestorben|nam|kind|vater|mutter|paten|braut|bräutigam",
                      re.I)
PLACE_HDR = re.compile(r"\bort\b|wohnort|haus|opuntali", re.I)
OCC_HDR = re.compile(r"krankheit|todesart|stand|beschäftig|gewerbe", re.I)


def field_type_for_column(header: str) -> str:
    h = header or ""
    if NAME_HDR.search(h):
        return "name"
    if PLACE_HDR.search(h):
        return "place"
    if OCC_HDR.search(h):
        return "occupation"
    return "name"   # default: treat as a name column (this corpus is name-dense)


# ── cell / sheet / workbook correction ───────────────────────────────────────
def correct_text(text: str, dic: TownDictionary, field_type: str,
                 sim_floor: float, log: list[dict]) -> str:
    def fix(m: re.Match) -> str:
        d = rerank_token(m.group(0), dic, field_type, sim_floor=sim_floor)
        if d.replaced:
            log.append({"from": d.token, "to": d.chosen,
                        "score": round(d.score, 1), "sim": round(d.sim, 1),
                        "field": field_type})
            return d.chosen
        return m.group(0)
    return WORD_RE.sub(fix, text)


def correct_xlsx(path: Path, dic: TownDictionary,
                 sim_floor: float = SIM_FLOOR) -> tuple[Path, list[dict]]:
    """Re-rank every cell of a Pernegg xlsx; write a corrected copy + log into
    OUT_DIR. The source file is never modified."""
    OUT_DIR.mkdir(exist_ok=True)
    xl = pd.ExcelFile(path)
    log: list[dict] = []
    # namespace by register folder so identical page names don't collide
    stem = f"{path.parent.name}__{path.stem}".replace(" ", "_")
    out_path = OUT_DIR / f"{stem}__dictguided.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        for sheet in xl.sheet_names:
            df = xl.parse(sheet, header=None)
            headers = [str(df.iloc[0, c]) if df.shape[0] else ""
                       for c in range(df.shape[1])]
            col_field = [field_type_for_column(h) for h in headers]
            out = df.copy()
            for r in range(df.shape[0]):
                for c in range(df.shape[1]):
                    val = df.iloc[r, c]
                    if pd.isna(val) or not isinstance(val, str):
                        continue
                    ft = col_field[c] if r > 0 else "skip"
                    if ft == "skip":
                        continue
                    out.iloc[r, c] = correct_text(val, dic, ft, sim_floor, log)
            out.to_excel(writer, sheet_name=sheet, header=False, index=False)
    (OUT_DIR / f"{stem}__replacements.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path, log


# ── OOV-segmented evaluation ─────────────────────────────────────────────────
def _normalise(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).replace("ſ", "s")
    return re.sub(r"\s+", " ", s).strip()


def _xlsx_to_text(path: Path) -> str:
    chunks = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None)
        for _, row in df.iterrows():
            cells = [_normalise(c) for c in row if pd.notna(c)]
            if cells:
                chunks.append(" ".join(cells))
    return "\n".join(chunks)


def _load_gt_names(gt_path: Path) -> list[dict]:
    """Ground-truth NAME fields only (the surname-OOV question)."""
    out = []
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = [_normalise(p).replace("(?)", "").strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[1] and parts[1] != "-":
            out.append({"record": parts[0], "gt": parts[1]})
    return out


def _aligned(gt: str, text: str) -> str:
    res = fuzz.partial_ratio_alignment(gt, text, processor=_normalise)
    return text[res.dest_start:res.dest_end] if res else ""


def _surname_of(name_field: str) -> str:
    """Heuristic: Austrian registers write 'GivenName Surname', so the surname
    is the LAST capitalised token (e.g. 'Elisabeth Sommer' -> 'Sommer',
    '... des Franz Reiter' -> 'Reiter')."""
    caps = WORD_RE.findall(name_field)
    return caps[-1] if caps else name_field


def eval_oov(gt_path: Path, xlsx: Path, dic: TownDictionary,
             taus: list[float]) -> dict:
    """Report CER/WER on NAME fields, split into in-lexicon vs OOV surnames,
    before and after dict-guided re-ranking, sweeping the τ (SIM_FLOOR)."""
    gt = _load_gt_names(gt_path)
    for e in gt:
        sn = _surname_of(e["gt"])
        e["surname"] = sn
        # three buckets keyed to the user's hypothesis: RTB should help surnames
        # that recur *in the region*; national-only is a weaker prior; OOV is
        # unreachable by any dictionary method.
        if sn in dic.region_count or sn in dic.town_surnames:
            e["bucket"] = "region"
        elif sn in dic.national_surnames:
            e["bucket"] = "national"
        else:
            e["bucket"] = "OOV"

    raw_text = _xlsx_to_text(xlsx)

    def score(entities, text):
        text = _normalise(text)
        cn = cd = wn = wd = 0
        for e in entities:
            span = _normalise(_aligned(e["gt"], text))
            cn += editdistance.eval(span, e["gt"]); cd += len(e["gt"])
            wn += editdistance.eval(span.split(), e["gt"].split())
            wd += len(e["gt"].split())
        return {"cer": cn / max(1, cd), "wer": wn / max(1, wd), "n": len(entities)}

    seg = {"all": gt,
           "region": [e for e in gt if e["bucket"] == "region"],
           "national": [e for e in gt if e["bucket"] == "national"],
           "OOV": [e for e in gt if e["bucket"] == "OOV"]}
    report = {"counts": {k: len(v) for k, v in seg.items()},
              "surnames": {e["surname"]: e["bucket"] for e in gt},
              "baseline": {k: score(v, raw_text) for k, v in seg.items()},
              "sweep": []}

    for tau in taus:
        log: list[dict] = []
        # correct the name-bearing text directly (string mode, no xlsx write)
        corrected = correct_text(raw_text, dic, "name", tau, log)
        row = {"tau": tau, "replacements": len(log),
               **{k: score(v, corrected) for k, v in seg.items()}}
        report["sweep"].append(row)
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────
def _print_eval(rep: dict):
    c = rep["counts"]
    print(f"GT name fields: all={c['all']}  region={c['region']}  "
          f"national={c['national']}  OOV={c['OOV']}\n")
    b = rep["baseline"]
    print("BASELINE (no re-ranking)")
    for seg in ("all", "region", "national", "OOV"):
        if c[seg]:
            print(f"  {seg:<9} CER={b[seg]['cer']:.3f}  WER={b[seg]['wer']:.3f}")
    print(f"\nτ-sweep (SIM_FLOOR)  [W_RTB={W_RTB} W_FREQ={W_FREQ}]")
    print(f"{'τ':>5} {'repl':>5} {'CER_all':>8} {'CER_region':>11} "
          f"{'CER_natl':>9} {'CER_OOV':>8}")
    for r in rep["sweep"]:
        print(f"{r['tau']:>5.0f} {r['replacements']:>5} "
              f"{r['all']['cer']:>8.3f} {r['region']['cer']:>11.3f} "
              f"{r['national']['cer']:>9.3f} {r['OOV']['cer']:>8.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("correct", help="re-rank a Pernegg xlsx -> corrected copy")
    pc.add_argument("--xlsx", required=True, type=Path)
    pc.add_argument("--parish", default="Pernegg (Steiermark)")
    pc.add_argument("--sim-floor", type=float, default=SIM_FLOOR)

    pb = sub.add_parser("batch", help="re-rank every Pernegg xlsx under a dir")
    pb.add_argument("--root", required=True, type=Path,
                    help="folder containing the per-register subfolders")
    pb.add_argument("--parish", default="Pernegg (Steiermark)")
    pb.add_argument("--sim-floor", type=float, default=SIM_FLOOR)

    pe = sub.add_parser("eval", help="OOV-segmented CER/WER, τ-sweep")
    pe.add_argument("--gt", required=True, type=Path)
    pe.add_argument("--xlsx", required=True, type=Path)
    pe.add_argument("--parish", default="Pernegg (Steiermark)")
    pe.add_argument("--taus", default="60,66,72,78,84",
                    help="comma-separated SIM_FLOOR values to sweep")
    pe.add_argument("--out", type=Path, default=OUT_DIR / "oov_eval.json")

    args = ap.parse_args()
    dic = load_town_dictionary(args.parish)
    print(f"parish={args.parish!r} -> town={dic.town!r} state={dic.state!r} | "
          f"1-hop={len(dic.town_surnames)} 2-hop(region)={len(dic.region_count)} "
          f"national={len(dic.national_surnames)}\n")

    if args.cmd == "correct":
        out_path, log = correct_xlsx(args.xlsx, dic, args.sim_floor)
        print(f"{len(log)} replacements -> {out_path}")
        for e in log[:40]:
            print(f"  [{e['field']}] {e['from']} -> {e['to']}  "
                  f"(sim={e['sim']} score={e['score']})")
    elif args.cmd == "batch":
        files = sorted(args.root.rglob("*.xlsx"))
        files = [f for f in files if not f.name.startswith("~$")]
        OUT_DIR.mkdir(exist_ok=True)
        manifest, total = [], 0
        for i, f in enumerate(files, 1):
            try:
                out_path, log = correct_xlsx(f, dic, args.sim_floor)
            except Exception as exc:                       # keep the batch going
                print(f"[{i}/{len(files)}] ERROR {f.parent.name}/{f.name}: {exc}")
                manifest.append({"src": str(f), "error": str(exc)})
                continue
            total += len(log)
            manifest.append({"src": str(f), "out": out_path.name,
                             "replacements": len(log)})
            print(f"[{i}/{len(files)}] {f.parent.name}/{f.name}: "
                  f"{len(log)} repl -> {out_path.name}")
        (OUT_DIR / "batch_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = sum(1 for m in manifest if "out" in m)
        print(f"\n{ok}/{len(files)} files corrected, {total} total replacements. "
              f"Source files untouched. Manifest: {OUT_DIR/'batch_manifest.json'}")
    else:
        taus = [float(x) for x in args.taus.split(",")]
        rep = eval_oov(args.gt, args.xlsx, dic, taus)
        OUT_DIR.mkdir(exist_ok=True)
        args.out.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        _print_eval(rep)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
