"""clean_surname_ocr.py — find and merge OCR misreadings in the KG v2 surname corpus.

Idea: a misreading is a RARE string that sits one OCR-typical edit away from a COMMON surname.
  1. Discovery: for every rare name (count <= RARE_MAX) find surnames at edit distance 1
     (plus a few known multi-char glyph confusions: rn<->m, ii<->ü, li<->h, cl<->d, ...)
     whose count is >= RATIO x larger. Tally the edit operations -> "usual suspects".
  2. Merge: accept variant -> canonical only if
       - the edit is a visual OCR confusion (OCR_OPS), NOT a genuine German spelling variant
         (ei/ey, i/y, dt/t, tz/z, ck/k, ph/f, ss/ß, th/t, c/k/z, ...), which existed side by side
         in 1900-1942 civil registers and must stay distinct (Meier/Meyer, Schmidt/Schmitt);
       - the canonical is >= RATIO x more frequent and attested in >= MIN_REGIONS regions;
       - the winner is unambiguous (best candidate >= 5x the runner-up).
Outputs (artifacts/kg_v2/):
  ocr_variant_map.tsv          variant, canonical, op, variant_count, canonical_count
  confusion_stats.tsv          op, n_variants, n_rows   (discovery, before filtering)
  surname_corpus_clean.tsv.gz  canonical surnames with merged counts
  ocr_audit_sample.tsv         60 random accepted merges for a manual check
  clean_stats.json
"""
import csv, gzip, json, random, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).parent / "artifacts" / "kg_v2"
RARE_MAX = 5          # only strings seen <= 5 times can be called misreadings
RATIO = 20            # canonical must be >= 20x more frequent
INDEL_RATIO = 50      # ...and >= 50x for a single inserted/deleted stroke
THIN = {"i", "l", "I", ".", "-", "'"}
MIN_REGIONS = 3       # canonical must be a widely attested name

# Visual confusions typical of OCR on 1940s Antiqua print (single-char substitutions a->b,
# meaning the PRINT had b and the OCR wrote a, i.e. variant char -> canonical char).
OCR_SUB = {
    ("u", "ü"), ("o", "ö"), ("a", "ä"), ("U", "Ü"), ("O", "Ö"), ("A", "Ä"),   # lost umlaut dots
    ("c", "e"), ("e", "c"), ("n", "u"), ("u", "n"), ("h", "b"), ("b", "h"),
    ("l", "t"), ("t", "l"), ("f", "t"), ("t", "f"), ("i", "l"), ("l", "i"), ("I", "l"),
    ("B", "ß"), ("ß", "B"), ("r", "n"), ("n", "r"), ("a", "o"), ("o", "a"), ("e", "o"),
    ("o", "e"), ("s", "e"), ("v", "y"), ("y", "v"), ("m", "n"), ("n", "m"), ("h", "k"),
    ("k", "h"), ("G", "C"), ("C", "G"), ("E", "F"), ("F", "E"), ("P", "R"), ("R", "P"),
    ("K", "R"), ("R", "K"), ("H", "N"), ("N", "H"), ("D", "O"), ("O", "D"), ("S", "8"),
}
# multi-char glyph confusions: variant substring -> canonical substring
OCR_MULTI = [("rn", "m"), ("m", "rn"), ("ii", "ü"), ("li", "h"), ("cl", "d"), ("ni", "m"),
             ("in", "m"), ("iu", "m"), ("vv", "w"), ("w", "vv"), ("ss", "ß"), ("fs", "ß"),
             ("Sch", "Sh")]
# genuine spelling variants: never merge across these (either direction)
SPELLING_PAIRS = {("i", "y"), ("y", "i"), ("t", "d"), ("d", "t"), ("z", "s"), ("s", "z"),
                  ("k", "c"), ("c", "k"), ("f", "v"), ("v", "f"), ("ss", "ß"), ("ß", "ss")}
GENUINE_SUBSTR = [("ei", "ey"), ("ai", "ay"), ("dt", "t"), ("tz", "z"), ("ck", "k"), ("ph", "f"),
                  ("th", "t"), ("mm", "m"), ("nn", "n"), ("ll", "l"), ("ff", "f"), ("tt", "t"),
                  ("rr", "r"), ("ss", "s"), ("ie", "i"), ("ah", "a"), ("eh", "e"), ("oh", "o"),
                  ("uh", "u"), ("dt", "d"), ("chs", "x"), ("ks", "x"), ("ae", "ä"), ("oe", "ö"),
                  ("ue", "ü"), ("ae", "a"), ("oe", "o"), ("ue", "u"), ("ee", "e"), ("aa", "a"),
                  ("oo", "o")]                 # old umlaut spellings (Haerder=Härder) and Low German doubles


def load():
    rows = list(csv.DictReader(gzip.open(D / "surname_corpus.tsv.gz", "rt", encoding="utf-8"), delimiter="\t"))
    return {r["surname"]: (int(r["count"]), int(r["n_regions"]), int(r["n_volumes"])) for r in rows}


def edits1(w):
    """(op, candidate) for every single-char sub/ins/del, plus multi-char glyph swaps."""
    out = []
    alpha = set("abcdefghijklmnopqrstuvwxyzäöüßABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜ-")
    for i in range(len(w)):
        out.append((f"del {w[i]}", w[:i] + w[i + 1:]))                 # OCR inserted w[i]
        for c in alpha:
            if c != w[i]:
                out.append((f"sub {w[i]}>{c}", w[:i] + c + w[i + 1:]))
    for i in range(len(w) + 1):
        for c in alpha - {"-"}:
            out.append((f"ins {c}", w[:i] + c + w[i:]))                # OCR dropped c
    for a, b in OCR_MULTI:
        start = 0
        while (j := w.find(a, start)) >= 0:
            out.append((f"glyph {a}>{b}", w[:j] + b + w[j + len(a):])); start = j + 1
    return out


def is_genuine_variant(v, c):
    for a, b in GENUINE_SUBSTR:
        for x, y in ((a, b), (b, a)):
            j = v.find(x)
            while j >= 0:
                if v[:j] + y + v[j + len(x):] == c:
                    return True
                j = v.find(x, j + 1)
    return False


def ocr_plausible(op):
    kind, rest = op.split(" ", 1)
    if kind == "glyph":
        a, b = rest.split(">")
        return (a, b) not in SPELLING_PAIRS
    if kind == "sub":
        a, b = rest.split(">")
        return (a, b) in OCR_SUB and (a, b) not in SPELLING_PAIRS
    if kind in ("del", "ins"):                 # only thin strokes appear/vanish in OCR; deleting or
        return rest in THIN                    # inserting e/n/r/t mostly merged REAL names (audit v1)
    return False


def main():
    corp = load()
    common = {s for s, (c, nr, _) in corp.items() if c >= RATIO and nr >= MIN_REGIONS}
    discovered = Counter(); disc_rows = Counter()
    accepted = {}
    for v, (vc, _, _) in corp.items():
        if vc > RARE_MAX or len(v) < 4:
            continue
        cands = []
        for op, c in edits1(v):
            # keep Bavarian/Austrian/Swabian diminutives (Mayrl, Zottl, Schönle) — real names
            if op == "del l" and (v.endswith("l") or v.endswith("le")) and not c.endswith(("l", "le")):
                continue
            if c in common and corp[c][0] >= RATIO * vc:
                discovered[op] += 1; disc_rows[op] += vc
                need = INDEL_RATIO if op.startswith(("del", "ins")) else RATIO
                if ocr_plausible(op) and not is_genuine_variant(v, c) and corp[c][0] >= need * vc:
                    cands.append((corp[c][0], c, op))
        if not cands:
            continue
        cands = sorted(set(cands), reverse=True)
        if len(cands) > 1 and cands[0][0] < 5 * cands[1][0]:
            continue                                         # ambiguous
        accepted[v] = (cands[0][1], cands[0][2])

    clean = Counter()
    for s, (c, _, _) in corp.items():
        clean[accepted[s][0] if s in accepted else s] += c

    with open(D / "ocr_variant_map.tsv", "w", encoding="utf-8") as fh:
        fh.write("variant\tcanonical\top\tvariant_count\tcanonical_count\n")
        for v, (c, op) in sorted(accepted.items(), key=lambda x: -corp[x[1][0]][0]):
            fh.write(f"{v}\t{c}\t{op}\t{corp[v][0]}\t{corp[c][0]}\n")
    with open(D / "confusion_stats.tsv", "w", encoding="utf-8") as fh:
        fh.write("op\tn_variants\tn_rows\tocr_plausible\tn_accepted\n")
        acc_ops = Counter(op for _, op in accepted.values())
        for op, n in discovered.most_common(300):
            fh.write(f"{op}\t{n}\t{disc_rows[op]}\t{ocr_plausible(op)}\t{acc_ops[op]}\n")
    with gzip.open(D / "surname_corpus_clean.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("surname\tcount\n")
        for s, c in clean.most_common():
            fh.write(f"{s}\t{c}\n")
    random.seed(0)
    sample = random.sample(sorted(accepted), min(60, len(accepted)))
    with open(D / "ocr_audit_sample.tsv", "w", encoding="utf-8") as fh:
        fh.write("variant\tcanonical\top\tvariant_count\tcanonical_count\tverdict\n")
        for v in sample:
            c, op = accepted[v]
            fh.write(f"{v}\t{c}\t{op}\t{corp[v][0]}\t{corp[c][0]}\t\n")

    st = {"surnames_before": len(corp), "surnames_after": len(clean),
          "variants_merged": len(accepted), "rows_relabelled": sum(corp[v][0] for v in accepted),
          "after_freq_ge": {k: sum(v >= k for v in clean.values()) for k in (2, 3, 5, 10)},
          "token_coverage_freq_ge": {k: round(sum(v for v in clean.values() if v >= k) / sum(clean.values()), 4)
                                     for k in (2, 3, 5, 10)},
          "top_accepted_ops": Counter(op for _, op in accepted.values()).most_common(25)}
    json.dump(st, open(D / "clean_stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(st, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
