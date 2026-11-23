"""Build the epoch-1 training subset from the full column manifest.

Selection: keep only columns that look like REAL subscriber listings (not
headers/footers/ads/legend fragments), then stratified-sample N examples
across volumes. Quality gates, per column:
  * n_entries >= --min-entries (default 8 — headers/ad cells are short)
  * >= --min-alpha share of surnames contain letters (kills number-only noise)
  * >= --min-ext share of entries carry a phone extension (the defining
    feature of a real directory line; legend/prose rows have none)
  * target length <= --max-target-chars (fits the A100 loss-memory budget)

Lexicon augmentation (--lexicon-mix): for that fraction of sampled examples,
append the town-scoped surname vocabulary from kg_town_surnames.json.gz to
the prompt. This trains DICTIONARY-GUIDED READING as a skill — the model
learns to resolve ambiguous strokes against a provided candidate list —
which generalises to any lexicon given at inference (new towns, and the
Kurrent parish registers this project ultimately targets), unlike pure
memorisation of corpus-frequent names.

Usage:
    venv/bin/python build_subset.py \
        --manifest /mnt/finetune/data_cols_full/manifest.jsonl \
        --out /mnt/finetune/data_cols_full/subset_10k.jsonl \
        --n 10000 --lexicon-mix 0.5 \
        --kg /mnt/finetune/kg_town_surnames.json.gz
"""
import argparse
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

LEXICON_BLOCK = """

VERIFIED SURNAMES ATTESTED IN THE TOWNS ON THIS COLUMN (from the 1942 \
directory; use them to resolve ambiguous letters, but never override what \
is clearly printed):
{names}
"""


def entry_quality(target: dict) -> tuple[int, float, float]:
    entries = [e for s in target["sections"] for e in s["entries"]]
    n = len(entries)
    if n == 0:
        return 0, 0.0, 0.0
    alpha = sum(1 for e in entries
                if any(c.isalpha() for c in e.get("surname", ""))) / n
    ext = sum(1 for e in entries if e.get("extension")) / n
    return n, alpha, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--min-entries", type=int, default=8)
    ap.add_argument("--min-alpha", type=float, default=0.7)
    ap.add_argument("--min-ext", type=float, default=0.6)
    ap.add_argument("--max-target-chars", type=int, default=14000)
    ap.add_argument("--lexicon-mix", type=float, default=0.5,
                    help="fraction of sampled examples that get the town-"
                         "scoped surname block appended to their prompt")
    ap.add_argument("--kg", type=Path, default=None,
                    help="kg_town_surnames.json.gz (town -> {surname: count})")
    ap.add_argument("--kg-top", type=int, default=80,
                    help="max surnames injected per example")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kg = {}
    if args.kg and args.lexicon_mix > 0:
        with gzip.open(args.kg, "rt", encoding="utf-8") as fh:
            kg = json.load(fh)
        print(f"KG loaded: {len(kg)} towns")

    by_vol = defaultdict(list)
    n_read = n_kept = 0
    for line in open(args.manifest, encoding="utf-8"):
        r = json.loads(line)
        n_read += 1
        if len(r["target"]) > args.max_target_chars:
            continue
        target = json.loads(r["target"])
        n, alpha, ext = entry_quality(target)
        if (n < args.min_entries or alpha < args.min_alpha
                or ext < args.min_ext):
            continue
        by_vol[r["volume"]].append(r)
        n_kept += 1
    print(f"quality filter: kept {n_kept}/{n_read}")
    for v in sorted(by_vol):
        print(f"  vol {v}: {len(by_vol[v])}")

    rng = random.Random(args.seed)
    per_vol = args.n // len(by_vol) if by_vol else 0
    sampled = []
    leftovers = []
    for v in sorted(by_vol):
        pool = by_vol[v]
        rng.shuffle(pool)
        sampled.extend(pool[:per_vol])
        leftovers.extend(pool[per_vol:])
    rng.shuffle(leftovers)
    sampled.extend(leftovers[:max(0, args.n - len(sampled))])
    rng.shuffle(sampled)
    print(f"sampled {len(sampled)} examples")

    n_aug = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for r in sampled:
            if kg and rng.random() < args.lexicon_mix:
                target = json.loads(r["target"])
                towns = {s["town"] for s in target["sections"] if s["town"]}
                names = {}
                for t in towns:
                    for s, c in kg.get(t, {}).items():
                        names[s] = max(names.get(s, 0), c)
                if names:
                    ranked = sorted(names, key=lambda s: -names[s])
                    block = LEXICON_BLOCK.format(
                        names=", ".join(ranked[:args.kg_top]))
                    r = dict(r)
                    r["prompt"] = r["prompt"].rstrip() + block
                    r["lexicon_augmented"] = True
                    n_aug += 1
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} ({n_aug} lexicon-augmented, "
          f"{len(sampled) - n_aug} plain)")


if __name__ == "__main__":
    main()
