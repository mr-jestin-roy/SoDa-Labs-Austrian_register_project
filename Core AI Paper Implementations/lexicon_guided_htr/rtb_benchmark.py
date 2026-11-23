"""Benchmark the lexicon-guided structured pipeline on Reichstelefonbuch pages.

Ground truth: the master_RTB_*.xlsx digitisation output (Chandra OCR runs,
human-spot-checked) keyed by page batch + cell crop filename; test images are
the column-section crops in Reichstelefonbuch_1942/cv_segment_cells (page
0749: Mattsee, Matzen, Mauerbach … — Austrian towns, printed Antiqua).

CAVEATS the numbers must be read with:
  * The RTB lexicon in artifacts/ was BUILT from these master files, so
    entities on the test page are largely in-vocabulary by construction.
    The OOV split below is therefore the honest signal: OOV entries are the
    hapaxes dropped by MIN_*_FREQ filtering plus master-side OCR errors.
  * The master itself contains OCR noise (e.g. page 0749 has
    "Zallhof Stiftesathaus" where the image reads "Zellhof, Stiftsgasthaus"),
    so a model can be right and still scored wrong — treat absolute scores
    as lower bounds and draft→final DELTAS as the reliable comparison.

Usage:
    venv/bin/python rtb_benchmark.py --cells col1_sec02.png col1_sec05.png \
        --model mlx-community/Qwen3-VL-4B-Instruct-4bit --out rtb_bench.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import config
from evaluate import cer, multiset_prf, strip_field
from kg_retrieval import LexiconKG
from pipeline import AUXILIARY_VOCAB, MASK_RE, parse_structured

RTB_DIR = config.PROJECT_ROOT / "Reichstelefonbuch_1942"
CELLS_DIR = RTB_DIR / "cv_segment_cells"
MASTER_XLSX = (RTB_DIR / "RTB_results_1_2_3_4"
               / "master_RTB_4_Karbow-Quittainen.xlsx")
PAGE_BATCH_PREFIX = "page_0749"

FIELDS = ("surname", "given_name", "profession")

# Telephone-book variant of the frozen register schema: same field names,
# same ⟦⟧ uncertainty convention, so FIELD_LEXICON_MAP and the evaluate.py
# multiset machinery apply unchanged.
RTB_PASS1_PROMPT = """\
You are an expert reader of the 1942 German Reichstelefonbuch (printed \
telephone directory, Antiqua typeface with bold surnames).

This image is one section of a directory column: a bold town header \
(possibly with the state in parentheses) followed by one subscriber per \
line: phone number, then name, then profession/business, sometimes a street.

Transcribe it as JSON with exactly this structure:

{
  "town": "<town header verbatim, e.g. 'Mattsee (Salzburg)'>",
  "entries": [
    {
      "extension": "<phone number as printed>",
      "surname": "<family name or institution name (first bold word(s))>",
      "given_name": "<given name(s) or initials if present, e.g. 'Johann', \
'Dr. Franz', 'Jak.'>",
      "profession": "<occupation/business if present, e.g. 'Fleischerei', \
'Arzt', 'Gasthof'>",
      "raw": "<the complete line verbatim>"
    }
  ]
}

Rules:
1. One entry per printed line. Omit a field entirely when the line does not \
give it (institutions like 'Gendarmerie' have only a surname field).
2. Keep original spelling and abbreviations verbatim; do not modernise.
3. Do NOT guess. Wrap any smudged or ambiguous reading in ⟦double brackets⟧ \
inside the field value, e.g. "surname": "⟦L?gl⟧".
4. Output ONLY the JSON object — no Markdown fences, no commentary.
"""

RTB_PASS2_TEMPLATE = """\
You are an expert reader of the 1942 German Reichstelefonbuch. You \
previously produced the DRAFT JSON below for a directory section from \
{town}; uncertain words are wrapped in ⟦double brackets⟧.

Re-examine the image and resolve every ⟦masked⟧ word by matching the printed \
letters against this verified vocabulary from the same directory region:

SURNAMES ATTESTED IN {town} ITSELF (most likely):
{town_surnames}

SURNAMES FROM THE SURROUNDING STATE ({state}):
{state_surnames}

COMMON GIVEN NAMES:
{given_names}

COMMON OCCUPATIONS/BUSINESS TYPES:
{occupations}

CALENDAR TERMS AND FORMULAE:
{auxiliary}
{per_token_block}
Rules:
1. A masked word may only be replaced by a vocabulary entry if it matches \
the printed letters. The vocabulary guides reading — it must not override \
the pixels.
2. If no vocabulary entry is visually plausible, transcribe what you see \
and keep the ⟦brackets⟧.
3. Fields that were already certain in the draft must be copied unchanged.
4. Output ONLY the corrected JSON object with the same structure — no \
Markdown fences, no commentary.

DRAFT:
{draft}
"""


# cv_segment_cells/*.png is a flat directory produced by a DIFFERENT
# segmentation run than the one behind master_RTB_*.xlsx (whose "cells"
# column restarts col1_secNN numbering per "batch", e.g. page_0749_b02,
# _b12 …). Filenames therefore collide across unrelated crops — verified by
# hand: master's cells=="col1_sec04.png" for page_0749 is "Matzen", but the
# actual image at cv_segment_cells/col1_sec04.png shows "Matzleinsdorf".
# The reliable join key is the TOWN HEADER text: each crop is exactly one
# town section, and master's "region" column carries the identical string.
# CELL_TOWN_MAP below records pairs verified by visually reading the crop
# and matching it to a distinct master "region" value for page_0749.
CELL_TOWN_MAP = {
    "col1_sec02.png": "Mattsee (Salzburg)",
    "col1_sec03.png": "Matzen (Niederdonau)",
    "col1_sec04.png": "Matzleinsdorf \xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0"
                      "\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0 b. Melk(Niederdonau)",
    "col1_sec05.png": "Mauer-Öhling (Niederdonau)",
    "col1_sec06.png": "Mauerbach (Niederdonaau)",  # sic — master's own typo
}


def load_ground_truth(cells: list[str]) -> dict[str, dict]:
    """cell filename -> {"town": str, "entries": [{field: value}]}, joined
    on verified town header text (see CELL_TOWN_MAP), NOT on the "cells"
    filename column, which does not correspond to this crop directory."""
    import pandas as pd
    df = pd.read_excel(MASTER_XLSX)
    df = df[df["batch"].astype(str).str.startswith(PAGE_BATCH_PREFIX)]
    out = {}
    for cell in cells:
        town = CELL_TOWN_MAP.get(cell)
        if town is None:
            print(f"no verified town mapping for {cell} — add it to "
                  f"CELL_TOWN_MAP after visually checking the crop",
                  file=sys.stderr)
            continue
        rows = df[df["region"].astype(str) == town]
        entries = []
        for _, r in rows.iterrows():
            text = str(r["transcribed_text"]).strip()
            if not text or text == "nan":
                continue
            entries.append(parse_gt_line(text, str(r["phone_extension"])))
        if entries:
            out[cell] = {"town": town, "entries": entries}
    return out


def parse_gt_line(text: str, extension: str) -> dict:
    """'Hueber, Dr. Franz, Justizminister' -> field dict (same shape the
    model emits, so the comparison is symmetric)."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    e = {"raw": text, "extension": extension.removesuffix(".0")}
    if parts:
        e["surname"] = parts[0]
    if len(parts) == 2:
        e["profession"] = parts[1]
    elif len(parts) >= 3:
        e["given_name"] = parts[1]
        e["profession"] = ", ".join(parts[2:])
    return e


def bags(entries: list[dict]) -> dict[str, Counter]:
    return {f: Counter(strip_field(e[f]) for e in entries if e.get(f))
            for f in FIELDS}


def oov_split(gold_entries: list[dict], pred_entries: list[dict],
              lex_words: set[str], field: str) -> dict:
    """Recall on gold values split by lexicon membership (first word of the
    value decides, matching how the lexicon stores single surnames)."""
    pred = Counter(strip_field(e[field]) for e in pred_entries if e.get(field))
    stats = {"in_vocab": [0, 0], "oov": [0, 0]}  # [hit, total]
    for e in gold_entries:
        v = strip_field(e.get(field, ""))
        if not v:
            continue
        bucket = "in_vocab" if v.split()[0] in lex_words else "oov"
        stats[bucket][1] += 1
        if pred[v] > 0:
            stats[bucket][0] += 1
            pred[v] -= 1
    return {k: {"recall": h / t if t else None, "n": t}
            for k, (h, t) in stats.items()}


def build_pass2(draft: str, scoped, kg: LexiconKG) -> str:
    masked = MASK_RE.findall(draft)
    per_token = ""
    if masked:
        aux_pool = [w.strip() for w in AUXILIARY_VOCAB.split(",")]
        lines = []
        for tok in dict.fromkeys(masked):
            cands = kg.candidates_for(tok.replace("?", "").replace("...", ""),
                                      scoped, extra_pool=aux_pool)
            if cands:
                lines.append(f"  ⟦{tok}⟧ → " + ", ".join(c for c, _ in cands))
        if lines:
            per_token = ("\nBEST CANDIDATES PER MASKED WORD (fuzzy-ranked):\n"
                         + "\n".join(lines) + "\n")
    return RTB_PASS2_TEMPLATE.format(
        town=scoped.matched_town or scoped.parish,
        state=scoped.state or "the region",
        town_surnames=", ".join(scoped.town_surnames) or "(none)",
        state_surnames=", ".join(scoped.state_surnames) or "(none)",
        given_names=", ".join(scoped.given_names),
        occupations=", ".join(scoped.occupations),
        auxiliary=AUXILIARY_VOCAB,
        per_token_block=per_token,
        draft=draft,
    )


# ── Whole-page mode ──────────────────────────────────────────────────────────
# A full printed page holds many town sections (e.g. RTB_5 p.6: Raab, Raabs,
# Raase, Raben, Raben-Steinfeld, Rabensburg, Rabenstein x2, Rabishau,
# Rackschütz, Rackwitz, Raboldshausen — scattered across five different
# German states). Ground truth here comes from a page-scoped OCR xlsx
# (RTB_results_1_2_3_4/results_RTB_5_Raab-Zygry/page_0006_ocr.xlsx) whose
# "region" column IS a safe join key because the file is already scoped to
# one page — unlike the flat cv_segment_cells crops used in cell mode.
PAGE_PASS1_PROMPT = """\
You are an expert reader of the 1942 German Reichstelefonbuch (printed \
telephone directory, Antiqua typeface, dense multi-column layout, bold \
town headers).

This image is a FULL PAGE containing multiple town sections, each starting \
with a bold town name (possibly with state in parentheses/brackets), \
followed by subscriber lines: phone number, name, profession/business, \
sometimes a street or place.

Transcribe the whole page as JSON:

{
  "sections": [
    {
      "town": "<town header verbatim, e.g. 'Raab (Oberdonau)'>",
      "entries": [
        {
          "extension": "<phone number as printed>",
          "surname": "<family name or institution name>",
          "given_name": "<given name(s)/initials if present>",
          "profession": "<occupation/business if present>",
          "raw": "<the complete line verbatim>"
        }
      ]
    }
  ]
}

Rules:
1. One section per town header, in reading order (columns left to right, \
top to bottom within each column). One entry per printed line.
2. Ignore running page headers/footers and the abbreviations legend line \
at the top of the page — they are not subscriber entries.
3. Omit a field entirely when the line does not give it.
4. Keep original spelling and abbreviations verbatim; do not modernise.
5. Do NOT guess. Wrap any smudged or ambiguous reading in ⟦double brackets⟧ \
inside the field value.
6. Output ONLY the JSON object — no Markdown fences, no commentary.
"""

PAGE_PASS2_TEMPLATE = """\
You are an expert reader of the 1942 German Reichstelefonbuch. You \
previously produced the DRAFT JSON below for a full page with multiple \
town sections; uncertain words are wrapped in ⟦double brackets⟧.

Re-examine the image and resolve every ⟦masked⟧ word by matching the \
printed letters against this verified vocabulary (drawn from the \
Reichstelefonbuch corpus generally — the page covers many different towns, \
so treat this as a broad prior, not a per-town restriction):

FREQUENT SURNAMES (nationwide sample):
{town_surnames}

MORE SURNAMES:
{state_surnames}

COMMON GIVEN NAMES:
{given_names}

COMMON OCCUPATIONS/BUSINESS TYPES:
{occupations}
{per_token_block}
Rules:
1. A masked word may only be replaced by a vocabulary entry if it matches \
the printed letters. The vocabulary guides reading — it must not override \
the pixels.
2. If no vocabulary entry is visually plausible, transcribe what you see \
and keep the ⟦brackets⟧.
3. Fields that were already certain in the draft must be copied unchanged.
4. Output ONLY the corrected JSON object with the same structure — no \
Markdown fences, no commentary.

DRAFT:
{draft}
"""


def load_page_ground_truth(xlsx_path: Path) -> dict:
    """Page-scoped OCR xlsx -> {"sections": [{"town", "entries"}]}. Safe to
    join on "region" because the file already belongs to a single page."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    sections, order = {}, []
    for _, r in df.iterrows():
        text = str(r["transcribed_text"]).strip()
        region = str(r["region"]).strip()
        if not text or text == "nan" or not region or region == "nan":
            continue
        # drop running headers / the abbreviations legend (long prose,
        # no town-like shape)
        if not re.fullmatch(r"[^,:]{2,60}(\s*[\(\[][^)\]]+[\)\]])?", region):
            continue
        if region not in sections:
            sections[region] = []
            order.append(region)
        sections[region].append(parse_gt_line(text, str(r["phone_extension"])))
    return {"sections": [{"town": t, "entries": sections[t]} for t in order
                         if sections[t]]}


def build_page_pass2(draft: str, scoped, kg: LexiconKG) -> str:
    masked = MASK_RE.findall(draft)
    per_token = ""
    if masked:
        aux_pool = [w.strip() for w in AUXILIARY_VOCAB.split(",")]
        lines = []
        for tok in dict.fromkeys(masked):
            cands = kg.candidates_for(tok.replace("?", "").replace("...", ""),
                                      scoped, extra_pool=aux_pool)
            if cands:
                lines.append(f"  ⟦{tok}⟧ → " + ", ".join(c for c, _ in cands))
        if lines:
            per_token = ("\nBEST CANDIDATES PER MASKED WORD (fuzzy-ranked):\n"
                         + "\n".join(lines) + "\n")
    return PAGE_PASS2_TEMPLATE.format(
        town_surnames=", ".join(scoped.town_surnames) or "(none)",
        state_surnames=", ".join(scoped.state_surnames) or "(none)",
        given_names=", ".join(scoped.given_names),
        occupations=", ".join(scoped.occupations),
        per_token_block=per_token,
        draft=draft,
    )


HTML_BLOCK_RE = re.compile(
    r'<div data-bbox="[^"]*" data-label="(?P<label>[^"]+)"[^>]*>'
    r'(?P<body>.*?)</div>', re.DOTALL)
LI_RE = re.compile(r"<li>(.*?)</li>", re.DOTALL)
P_RE = re.compile(r"<p>(.*?)</p>", re.DOTALL)
LEADING_EXT_RE = re.compile(r"^\s*(\d+|\.\.|-)\s*")
TAG_RE = re.compile(r"<[^>]+>")


def parse_chandra_html(text: str) -> dict | None:
    """Chandra OCR 2 ignores custom JSON schemas for full-page transcription
    and reverts to its own trained layout format: bbox-annotated HTML with
    "Section-Header" (town name) and "List-Group" (<li> containing several
    newline-joined subscriber lines) — the same convention as Qwen3-VL's
    "QwenVL-HTML" fine-grained layout output. Rather than fight this, parse
    it natively into the same {"sections": [...]} shape as the JSON path."""
    blocks = HTML_BLOCK_RE.findall(text)
    if not blocks:
        return None
    sections, current_town, current_entries = [], None, []

    def flush():
        if current_town is not None and current_entries:
            sections.append({"town": current_town, "entries": current_entries})

    for label, body in blocks:
        if label == "Section-Header":
            flush()
            m = P_RE.search(body)
            current_town = TAG_RE.sub("", m.group(1)).strip() if m else ""
            current_entries = []
        elif label == "List-Group" and current_town is not None:
            for li in LI_RE.findall(body):
                raw_lines = TAG_RE.sub("", li).splitlines()
                for line in raw_lines:
                    line = line.strip()
                    if not line:
                        continue
                    ext_m = LEADING_EXT_RE.match(line)
                    ext = ext_m.group(1) if ext_m else ""
                    rest = line[ext_m.end():].strip() if ext_m else line
                    if rest:
                        current_entries.append(parse_gt_line(rest, ext))
    flush()
    return {"sections": sections} if sections else None


def normalize_page_pred(obj) -> dict | None:
    """Same field-name/shape tolerance as normalize_pred, but for the
    {"sections": [...]} page-level envelope."""
    if obj is None:
        return None
    if isinstance(obj, list):
        obj = {"sections": obj}
    if not isinstance(obj, dict) or not isinstance(obj.get("sections"), list):
        return None
    fixed = []
    for sec in obj["sections"]:
        if not isinstance(sec, dict):
            continue
        norm = normalize_pred(sec if isinstance(sec.get("entries"), list)
                              else {"town": sec.get("town", ""), "entries": []})
        fixed.append(norm or {"town": sec.get("town", ""), "entries": []})
    return {"sections": fixed}


def score_page(pred: dict | None, gold: dict, lex: dict) -> dict:
    pred = normalize_page_pred(pred)
    if not pred:
        return {"parsed": False}
    pe = [e for sec in pred["sections"] for e in sec.get("entries", [])
          if isinstance(e, dict)]
    ge = [e for sec in gold["sections"] for e in sec["entries"]]
    s = {"parsed": True, "n_pred_sections": len(pred["sections"]),
         "n_gold_sections": len(gold["sections"]),
         "n_pred_entries": len(pe), "n_gold_entries": len(ge), "fields": {}}
    pb, gb = bags(pe), bags(ge)
    for f in FIELDS:
        p, r, f1 = multiset_prf(pb.get(f, Counter()), gb.get(f, Counter()))
        s["fields"][f] = {"precision": round(p, 3), "recall": round(r, 3),
                          "f1": round(f1, 3)}
    s["oov"] = {
        "surname": oov_split(ge, pe, set(lex["surnames"]), "surname"),
        "profession": oov_split(ge, pe, set(lex["occupations"]), "profession"),
    }
    gold_raw = " ".join(sorted(e["raw"] for e in ge))
    pred_raw = " ".join(sorted(strip_field(str(e.get("raw", ""))) for e in pe))
    s["raw_cer_sorted"] = round(cer(pred_raw, gold_raw), 4)
    return s


def parse_page_output(text: str) -> dict | None:
    """Try the requested JSON schema first; fall back to Chandra's native
    bbox-HTML layout format if the model ignored the schema (empirically,
    it does for full-page transcription — see parse_chandra_html)."""
    parsed, _ = parse_structured(text)
    parsed = normalize_page_pred(parsed)
    if parsed and parsed["sections"]:
        return parsed
    return parse_chandra_html(text)


def run_page(image: Path, truth_xlsx: Path, model: str, iterations: int,
            scope_town: str, out: Path, max_tokens: int = 16000):
    gold = load_page_ground_truth(truth_xlsx)
    n_entries = sum(len(sec["entries"]) for sec in gold["sections"])
    print(f"Ground truth: {len(gold['sections'])} sections, "
          f"{n_entries} entries — "
          + ", ".join(sec["town"] for sec in gold["sections"]),
          file=sys.stderr)

    kg = LexiconKG.load()
    scoped = kg.scoped_lexicon(scope_town)
    from vlm_client import VLMClient
    client = VLMClient(model=model)

    draft = client.transcribe(image, PAGE_PASS1_PROMPT, max_tokens=max_tokens)
    parsed_draft = parse_page_output(draft)
    rec = {"draft_masks": len(MASK_RE.findall(draft)),
           "draft": score_page(parsed_draft, gold, kg.lex), "passes": [draft]}
    current = draft
    for i in range(iterations):
        if not MASK_RE.search(current) and i > 0:
            break
        current = client.transcribe(image, build_page_pass2(current, scoped, kg),
                                    max_tokens=max_tokens)
        rec["passes"].append(current)
    parsed_final = parse_page_output(current)
    rec["final_masks"] = len(MASK_RE.findall(current))
    rec["final"] = score_page(parsed_final, gold, kg.lex)

    for label in ("draft", "final"):
        s = rec[label]
        if s.get("parsed"):
            fs = " ".join(f"{f}:F1={s['fields'][f]['f1']}" for f in FIELDS)
            print(f"{label}: {fs} raw-CER={s['raw_cer_sorted']} "
                  f"sections={s['n_pred_sections']}/{s['n_gold_sections']} "
                  f"entries={s['n_pred_entries']}/{s['n_gold_entries']} "
                  f"oov-surname={s['oov']['surname']['oov']} "
                  f"in-vocab-surname={s['oov']['surname']['in_vocab']}",
                  file=sys.stderr)
        else:
            print(f"{label}: JSON unparseable", file=sys.stderr)

    report = {"model": model, "mode": "page", "image": str(image),
             "truth_xlsx": str(truth_xlsx), **rec}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nWrote {out}", file=sys.stderr)


def normalize_pred(obj) -> dict | None:
    """Chandra OCR 2 does not reliably follow the custom RTB schema: it
    tends to emit a bare list (not {"town", "entries"}) and merge given_name
    + surname into one "name" field. Adapt its native output into our shape
    using the same comma-split heuristic as parse_gt_line, so the two are
    scored on equal footing instead of failing to parse at all."""
    if obj is None:
        return None
    if isinstance(obj, list):
        obj = {"town": "", "entries": obj}
    if not isinstance(obj, dict) or not isinstance(obj.get("entries"), list):
        return None
    fixed_entries = []
    for e in obj["entries"]:
        if not isinstance(e, dict):
            continue
        e = dict(e)
        name = e.pop("name", None)
        if name and not e.get("surname"):
            parts = [p.strip() for p in str(name).split(",") if p.strip()]
            if parts:
                e["surname"] = parts[0]
            if len(parts) >= 2 and not e.get("given_name"):
                e["given_name"] = parts[1]
            if len(parts) >= 3 and not e.get("profession"):
                e["profession"] = ", ".join(parts[2:])
        fixed_entries.append(e)
    obj["entries"] = fixed_entries
    return obj


def score(pred: dict | None, gold: dict, lex: dict) -> dict:
    pred = normalize_pred(pred)
    if not pred or not isinstance(pred.get("entries"), list):
        return {"parsed": False}
    pe, ge = pred["entries"], gold["entries"]
    pe = [e for e in pe if isinstance(e, dict)]
    s = {"parsed": True, "n_pred": len(pe), "n_gold": len(ge),
         "town_ok": strip_field(str(pred.get("town", ""))) == gold["town"],
         "fields": {}}
    pb, gb = bags(pe), bags(ge)
    for f in FIELDS:
        p, r, f1 = multiset_prf(pb.get(f, Counter()), gb.get(f, Counter()))
        s["fields"][f] = {"precision": round(p, 3), "recall": round(r, 3),
                          "f1": round(f1, 3)}
    s["oov"] = {
        "surname": oov_split(ge, pe, set(lex["surnames"]), "surname"),
        "profession": oov_split(ge, pe, set(lex["occupations"]), "profession"),
    }
    gold_raw = " ".join(sorted(e["raw"] for e in ge))
    pred_raw = " ".join(sorted(strip_field(str(e.get("raw", "")))
                               for e in pe))
    s["raw_cer_sorted"] = round(cer(pred_raw, gold_raw), 4)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--page-image", type=Path, default=None,
                    help="whole rendered PDF page (see rtb_benchmark.py "
                         "module docstring: 'Whole-page mode')")
    ap.add_argument("--page-truth", type=Path, default=None,
                    help="matching page_NNNN_ocr.xlsx ground truth")
    ap.add_argument("--page-scope-town", default=config.DEFAULT_PARISH,
                    help="dominant town on the page, used to seed the "
                         "pass-2 surname prior (global fallback still "
                         "applies via LexiconKG.candidates_for)")
    ap.add_argument("--model",
                    default="mlx-community/Qwen3-VL-4B-Instruct-4bit")
    ap.add_argument("--iterations", type=int, default=1,
                    help="lexicon-guided refinement passes after the draft")
    ap.add_argument("--out", type=Path, default=Path("rtb_bench.json"))
    args = ap.parse_args()

    if args.page_image or args.page_truth:
        if not (args.page_image and args.page_truth):
            sys.exit("--page-image and --page-truth must be given together")
        run_page(args.page_image, args.page_truth, args.model,
                 args.iterations, args.page_scope_town, args.out)
        return

    gt = load_ground_truth(args.cells or ["col1_sec02.png"])
    missing = [c for c in args.cells if c not in gt]
    if missing:
        print(f"no ground truth for {missing}, skipping", file=sys.stderr)

    kg = LexiconKG.load()
    from vlm_client import VLMClient
    client = VLMClient(model=args.model)

    report = {"model": args.model, "master": MASTER_XLSX.name, "cells": {}}
    for cell, gold in gt.items():
        image = CELLS_DIR / cell
        print(f"\n=== {cell} — {gold['town']} "
              f"({len(gold['entries'])} gold entries) ===", file=sys.stderr)
        scoped = kg.scoped_lexicon(gold["town"])
        draft = client.transcribe(image, RTB_PASS1_PROMPT)
        parsed_draft, _ = parse_structured(draft)
        rec = {"gold_town": gold["town"],
               "draft_masks": len(MASK_RE.findall(draft)),
               "draft": score(parsed_draft, gold, kg.lex),
               "passes": [draft]}
        current = draft
        for i in range(args.iterations):
            if not MASK_RE.search(current) and i > 0:
                break
            current = client.transcribe(image, build_pass2(current, scoped, kg))
            rec["passes"].append(current)
        parsed_final, _ = parse_structured(current)
        rec["final_masks"] = len(MASK_RE.findall(current))
        rec["final"] = score(parsed_final, gold, kg.lex)
        report["cells"][cell] = rec
        for label in ("draft", "final"):
            s = rec[label]
            if s.get("parsed"):
                fs = " ".join(f"{f}:F1={s['fields'][f]['f1']}" for f in FIELDS)
                print(f"{label}: {fs} raw-CER={s['raw_cer_sorted']} "
                      f"oov-surname-recall="
                      f"{s['oov']['surname']['oov']}", file=sys.stderr)
            else:
                print(f"{label}: JSON unparseable", file=sys.stderr)

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
