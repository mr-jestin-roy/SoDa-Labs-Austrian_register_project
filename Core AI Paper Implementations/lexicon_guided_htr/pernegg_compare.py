"""Compare three systems on the Pernegg Sterberegister test pages:

  prev     — the existing Chandra single-pass output (xlsx, read-only)
  baseline — pass 1 of the new pipeline (vision-only draft, masks stripped)
  guided   — final lexicon-guided output of the new pipeline

Metric: entity-level CER/WER. For every ground-truth entity string
(name, occupation, place, cause of death) we locate the best-matching
substring in each system's full output text via rapidfuzz alignment and
compute character/word edit distance of that aligned span against the GT.
This is layout-independent: it scores how well each system rendered the
*content*, not the Markdown table formatting.

Usage:
    venv/bin/python pernegg_compare.py \
        --gt pernegg_run/gt_seite_11-12_entities.txt \
        --prev-xlsx "/path/to/Seite 11-12.xlsx" \
        --result pernegg_run/result_seite_11-12.json
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import editdistance
import pandas as pd
from rapidfuzz import fuzz

MASK_RE = re.compile(r"⟦([^⟧]*)⟧")


def normalise(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("ſ", "s")
    return re.sub(r"\s+", " ", s).strip()


def load_gt(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = [normalise(p).replace("(?)", "").strip()
                 for p in line.split("|")]
        rec_id, fields = parts[0], parts[1:]
        labels = ["name", "occupation", "place", "cause"]
        for label, value in zip(labels, fields):
            if value and value != "-":
                records.append({"record": rec_id, "field": label,
                                "gt": value})
    return records


def xlsx_to_text(path: Path) -> str:
    chunks = []
    xl = pd.ExcelFile(path)
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None)
        for _, row in df.iterrows():
            cells = [normalise(c) for c in row if pd.notna(c)]
            if cells:
                chunks.append(" ".join(cells))
    return "\n".join(chunks)


def aligned_span(gt: str, text: str) -> str:
    """Best-matching substring of `text` for `gt` (rapidfuzz alignment)."""
    res = fuzz.partial_ratio_alignment(gt, text, processor=normalise)
    if res is None:
        return ""
    return text[res.dest_start:res.dest_end]


def score(entities: list[dict], text: str) -> dict:
    text = normalise(text)
    total_cer_num = total_cer_den = 0
    total_wer_num = total_wer_den = 0
    spans = []
    for e in entities:
        span = normalise(aligned_span(e["gt"], text))
        gt = e["gt"]
        total_cer_num += editdistance.eval(span, gt)
        total_cer_den += len(gt)
        total_wer_num += editdistance.eval(span.split(), gt.split())
        total_wer_den += len(gt.split())
        spans.append(span)
    return {"cer": total_cer_num / max(1, total_cer_den),
            "wer": total_wer_num / max(1, total_wer_den),
            "spans": spans}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--prev-xlsx", required=True, type=Path)
    ap.add_argument("--result", type=Path, default=None,
                    help="pipeline.py output JSON (passes[0]=baseline, final=guided)")
    ap.add_argument("--parish", default="Pernegg (Steiermark)")
    ap.add_argument("--details", action="store_true",
                    help="print per-entity aligned spans")
    args = ap.parse_args()

    entities = load_gt(args.gt)
    strip = lambda t: MASK_RE.sub(lambda m: m.group(1), t)

    prev_text = xlsx_to_text(args.prev_xlsx)
    systems = {"prev (old single-pass)": prev_text}

    # post-OCR fuzzy-correction arm: prev output + lexicon, no image access
    from kg_retrieval import LexiconKG
    from postocr_correct import correct_text
    from pipeline import AUXILIARY_VOCAB
    kg = LexiconKG.load()
    scoped = kg.scoped_lexicon(args.parish)
    aux = [w.strip() for w in AUXILIARY_VOCAB.split(",")]
    fixed, log = correct_text(prev_text, kg, scoped, aux)
    systems["prev + post-OCR fix"] = fixed

    if args.result:
        result = json.loads(args.result.read_text(encoding="utf-8"))
        systems["baseline (new pass 1)"] = strip(result["passes"][0]["text"])
        systems["guided (lexicon final)"] = strip(result["final"])

    print(f"{len(entities)} GT entity fields from {args.gt.name}")
    print(f"post-OCR arm made {len(log)} replacements: "
          + ", ".join(f"{e['from']}→{e['to']}" for e in log[:12]) + "\n")
    print(f"{'system':<26} {'CER':>8} {'WER':>8}")
    results = {}
    for name, text in systems.items():
        r = score(entities, text)
        results[name] = r
        print(f"{name:<26} {r['cer']:>8.3f} {r['wer']:>8.3f}")

    if args.details:
        print()
        keys = list(systems)
        for i, e in enumerate(entities):
            print(f"[{e['record']}:{e['field']}] GT: {e['gt']}")
            for k in keys:
                print(f"    {k.split()[0]:<10}: {results[k]['spans'][i]}")


if __name__ == "__main__":
    main()
