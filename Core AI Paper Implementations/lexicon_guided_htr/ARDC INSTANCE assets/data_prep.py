"""Build a LoRA fine-tuning corpus for Chandra OCR 2 from the digitised
Reichstelefonbuch 1942 (5 volumes). Each training example is one full PDF
page image paired with a gold JSON transcription in the frozen page-level
schema (matches rtb_benchmark.py's {"sections": [{"town", "entries"}]}
shape, so downstream lexicon-guided constrained decoding can reuse
FIELD_LEXICON_MAP unchanged).

Why JSON, not Chandra's native bbox-HTML format: prompting alone could not
get Chandra OCR 2 to emit our schema (it reverts to its own trained
"QwenVL-HTML"-style layout format — see rtb_benchmark.py's
parse_chandra_html). LoRA exists to close exactly this gap.

Ground truth source: RTB_results_1_2_3_4/results_RTB_<N>_<name>/
page_NNNN_ocr.xlsx, columns [batch, cells, region, phone_extension,
transcribed_text] — one row per printed subscriber line, "region" is the
town header (safe join key; see rtb_benchmark.py's CELL_TOWN_MAP note for
why filename-based joins are NOT safe elsewhere in this corpus).

Usage:
    venv/bin/python data_prep.py --rtb-dir /path/to/Reichstelefonbuch_1942 \
        --out-dir /mnt/finetune/data --volumes 1 2 3 4 5 \
        --max-pages-per-volume 30   # omit for the full corpus (~6622 pages)
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

VOLUMES = {
    1: "RTB_1_Aach-Berlichingen",
    2: "RTB_2_Berlin-Dotterwies",
    3: "RTB_3_Drachhausen-Kaprun",
    4: "RTB_4_Karbow-Quittainen",
    5: "RTB_5_Raab-Zygry",
}

# Identical instruction used at inference time in rtb_benchmark.py — MUST
# stay byte-identical between training and inference prompts (same coupling
# rule as schema.py's C3 note earlier in this project).
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


def parse_gt_line(text: str, extension: str) -> dict:
    """'Hueber, Dr. Franz, Justizminister' -> field dict. Ported verbatim
    from rtb_benchmark.py to keep this script dependency-free (no
    kg_retrieval/rapidfuzz needed on the training box)."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    e = {"raw": text}
    ext = str(extension).removesuffix(".0")
    if ext and ext != "nan":
        e["extension"] = ext
    if parts:
        e["surname"] = parts[0]
    if len(parts) == 2:
        e["profession"] = parts[1]
    elif len(parts) >= 3:
        e["given_name"] = parts[1]
        e["profession"] = ", ".join(parts[2:])
    return e


def load_page_ground_truth(xlsx_path: Path) -> dict:
    df = pd.read_excel(xlsx_path)
    sections, order = {}, []
    for _, r in df.iterrows():
        text = str(r["transcribed_text"]).strip()
        region = str(r["region"]).strip()
        if not text or text == "nan" or not region or region == "nan":
            continue
        if not re.fullmatch(r"[^,:]{2,60}(\s*[\(\[][^)\]]+[\)\]])?", region):
            continue
        if region not in sections:
            sections[region] = []
            order.append(region)
        sections[region].append(parse_gt_line(text, r["phone_extension"]))
    return {"sections": [{"town": t, "entries": sections[t]} for t in order
                         if sections[t]]}


def render_page(pdf_path: Path, page_num: int, out_png: Path, dpi: int = 200):
    if out_png.exists():
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    stem = out_png.with_suffix("")
    subprocess.run(
        ["pdftoppm", "-f", str(page_num), "-l", str(page_num),
         "-r", str(dpi), "-png", "-singlefile", str(pdf_path), str(stem)],
        check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtb-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--volumes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--max-pages-per-volume", type=int, default=None,
                    help="cap for quick pipeline validation runs; omit for "
                         "the full ~6622-page corpus")
    ap.add_argument("--min-entries", type=int, default=3,
                    help="skip pages with fewer than this many gold entries "
                         "(near-blank / front-matter / ad-only pages)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    images_dir = args.out_dir / "images"
    manifest_path = args.out_dir / "manifest.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_written = n_skipped = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for vol in args.volumes:
            name = VOLUMES[vol]
            pdf_path = args.rtb_dir / f"{name}.pdf"
            xlsx_dir = args.rtb_dir / "RTB_results_1_2_3_4" / f"results_{name}"
            xlsx_files = sorted(xlsx_dir.glob("page_*_ocr.xlsx"))
            if args.max_pages_per_volume:
                xlsx_files = xlsx_files[:args.max_pages_per_volume]
            print(f"vol {vol} ({name}): {len(xlsx_files)} candidate pages",
                  file=sys.stderr)

            for xlsx_path in xlsx_files:
                m = re.search(r"page_(\d+)_ocr\.xlsx", xlsx_path.name)
                page_num = int(m.group(1))
                gold = load_page_ground_truth(xlsx_path)
                n_entries = sum(len(s["entries"]) for s in gold["sections"])
                if n_entries < args.min_entries:
                    n_skipped += 1
                    continue

                png_path = images_dir / f"vol{vol}" / f"page_{page_num:04d}.png"
                try:
                    render_page(pdf_path, page_num, png_path, args.dpi)
                except subprocess.CalledProcessError as e:
                    print(f"  render failed vol{vol} page{page_num}: "
                          f"{e.stderr.decode(errors='replace')[:200]}",
                          file=sys.stderr)
                    n_skipped += 1
                    continue

                manifest.write(json.dumps({
                    "image": str(png_path),
                    "volume": vol, "page": page_num,
                    "prompt": PAGE_PASS1_PROMPT,
                    "target": json.dumps(gold, ensure_ascii=False),
                    "n_entries": n_entries,
                }, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"\nWrote {n_written} examples to {manifest_path} "
          f"({n_skipped} skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
