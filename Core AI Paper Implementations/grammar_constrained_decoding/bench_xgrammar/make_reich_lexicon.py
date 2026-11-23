"""make_reich_lexicon.py — Reich-wide name list for the G3_reich grammar condition.

names = OCR-cleaned KG v2 surnames seen >= 2 times (surname_corpus_clean.tsv.gz)
      + organisation entry starts seen >= 20 times that are not person surnames (Bahnhof …,
        Gasthof …, Gemeindeamt …), because ~40% of RTB entries start with an organisation.
Leak check: the bench pages 1266-1285 are part of the RTB corpus, so their own names are in the
list. To measure a FAIR closed-vocabulary condition, names whose ONLY attestations come from
pages 1266-1285 are dropped (they would not be in a list built from the other 1,874k rows).
Output: reich_names.txt (one per line), reich_lexicon_stats.json
"""
import csv, gzip, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KG = ROOT / "Core AI Paper Implementations" / "lexicon_guided_htr" / "artifacts" / "kg_v2"
REF = ROOT / "Core AI Paper Implementations" / "grammar_constrained_decoding" / "infra" / "data" / "rtb4_p1266_1285_reference.csv"
OUT = Path(__file__).parent
MIN_SURNAME, MIN_ORG = 2, 20

sur = {r["surname"]: int(r["count"]) for r in csv.DictReader(gzip.open(KG / "surname_corpus_clean.tsv.gz", "rt", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}
starts = {}
for line in gzip.open(KG / "entry_starts.tsv.gz", "rt", encoding="utf-8").read().splitlines()[1:]:
    if "\t" in line:
        k, c = line.rsplit("\t", 1)
        if c.isdigit():
            starts[k] = int(c)

# counts contributed by the bench pages themselves (first comma field of each reference row)
bench = Counter()
for r in csv.DictReader(open(REF, encoding="utf-8")):
    t = (r["transcribed_text"] or "").split(",")[0].strip(" .")
    if t:
        bench[t[:80]] += 1

names, dropped_leak = [], 0
for s, c in sur.items():
    if c - bench.get(s, 0) >= MIN_SURNAME:
        names.append(s)
    elif c >= MIN_SURNAME:
        dropped_leak += 1
person = set(sur)
orgs = [s for s, c in starts.items()
        if c - bench.get(s, 0) >= MIN_ORG and s not in person and not re.fullmatch(r"[\d\W]+", s)
        and 2 <= len(s) <= 60]
allnames = sorted(set(names) | set(orgs))
(OUT / "reich_names.txt").write_text("\n".join(allnames) + "\n", encoding="utf-8")
st = {"surnames": len(names), "orgs": len(orgs), "total": len(allnames), "dropped_bench_only": dropped_leak,
      "chars": sum(len(n) for n in allnames)}
json.dump(st, open(OUT / "reich_lexicon_stats.json", "w"), indent=1)
print(st); print("orgs sample:", orgs[:25])
