"""Build a CELL-LEVEL LoRA corpus for Chandra OCR 2 from the digitised
Reichstelefonbuch — replacing the full-page data_prep.py, whose page-sized
targets OOMed the loss (seq_len × vocab 248,320 fp32 logits) on a 40GB A100.

Reproduces the ORIGINAL digitisation pipeline (see
"digitizing python scripts/orchestrate_pipeline.py"):
    PDF page → 600 DPI PNG (PyMuPDF) → cv_segment → cell crops
so that each crop's name (col<C>_sec<S>.png) joins directly to the "cells"
column of that page's ground-truth xlsx. Segmentation parameters are ported
verbatim from cv_segment_test.py — same thresholds, kernels, margins — to
keep the crop↔xlsx correspondence faithful. Pages where the regenerated
cell-name set does not cover the xlsx's cells are SKIPPED (segmentation
drift), so no example can be silently mispaired (cf. the CELL_TOWN_MAP
lesson in rtb_benchmark.py).

Each training example: one cell crop + section-level JSON target
{"town": ..., "entries": [{extension, surname, given_name, profession,
raw}]} — the same shape as rtb_benchmark.py's cell-mode RTB_PASS1_PROMPT,
which is reused verbatim as the training instruction.

Usage (on the A100 box):
    venv/bin/python data_prep_cells.py --rtb-dir /mnt/finetune/rtb_source \
        --out-dir /mnt/finetune/data_cells --volumes 1 2 3 4 5 \
        --max-pages-per-volume 10   # omit for the full corpus
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VOLUMES = {
    1: "RTB_1_Aach-Berlichingen",
    2: "RTB_2_Berlin-Dotterwies",
    3: "RTB_3_Drachhausen-Kaprun",
    4: "RTB_4_Karbow-Quittainen",
    5: "RTB_5_Raab-Zygry",
}

DPI = 600  # orchestrate_pipeline.py's render DPI — do not change

# ── Segmentation params, ported VERBATIM from cv_segment_test.py ─────────────
ADAPTIVE_BLOCK = 21
ADAPTIVE_C = 10
V_LINE_MIN_LEN = 600
H_FRAC_OF_COL = 0.05
CLUSTER_GAP = 45
CROP_MARGIN = 18

# Identical to rtb_benchmark.py's cell-mode RTB_PASS1_PROMPT — training and
# inference instructions MUST stay byte-identical (schema.py C3 coupling rule).
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


def render_page_600dpi(pdf_path: Path, page_num: int) -> np.ndarray:
    """PDF page → BGR image at 600 DPI via PyMuPDF, matching
    orchestrate_pipeline.py's extraction (zoom = DPI/72)."""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    zoom = DPI / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    doc.close()
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cluster(positions, gap):
    if not positions:
        return []
    positions = sorted(set(positions))
    groups = [[positions[0]]]
    for p in positions[1:]:
        if p - groups[-1][-1] <= gap:
            groups[-1].append(p)
        else:
            groups.append([p])
    return [int(np.mean(g)) for g in groups]


def segment_cells(img: np.ndarray) -> dict[str, np.ndarray]:
    """cv_segment_test.py's cell detection → {"col1_sec01.png": crop, ...}"""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK, ADAPTIVE_C)

    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, V_LINE_MIN_LEN))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kern, iterations=2)
    v_xs_raw = [x for x in range(W) if v_lines[:, x].sum() > 0]
    col_splits = cluster(v_xs_raw, CLUSTER_GAP)
    x_bounds = [0] + col_splits + [W]

    cells = {}
    for ci in range(len(x_bounds) - 1):
        cx1, cx2 = x_bounds[ci], x_bounds[ci + 1]
        col_w = cx2 - cx1
        h_min_len = max(30, int(col_w * H_FRAC_OF_COL))
        col_thresh = thresh[:, cx1:cx2]
        h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (h_min_len, 1))
        h_lines_col = cv2.morphologyEx(col_thresh, cv2.MORPH_OPEN, h_kern,
                                       iterations=2)
        h_ys_raw = [y for y in range(H) if h_lines_col[y, :].sum() > 0]
        row_splits = cluster(h_ys_raw, CLUSTER_GAP)
        y_bounds = [0] + row_splits + [H]
        for si in range(len(y_bounds) - 1):
            y1 = y_bounds[si] + CROP_MARGIN
            y2 = y_bounds[si + 1] - CROP_MARGIN
            x1 = cx1 + CROP_MARGIN
            x2 = cx2 - CROP_MARGIN
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                cells[f"col{ci + 1}_sec{si + 1:02d}.png"] = img[y1:y2, x1:x2]
    return cells


def parse_gt_line(text: str, extension: str) -> dict:
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


TOWN_RE = re.compile(r"[^,:]{2,60}(\s*[\(\[][^)\]]+[\)\]])?")


def load_cells_ground_truth(xlsx_path: Path) -> dict[str, dict]:
    """{"col1_sec03.png": {"town", "entries"}} for one page's xlsx. The
    "cells" column is page-scoped here, so it IS a safe join key (unlike
    the flat cv_segment_cells directory — see rtb_benchmark.py)."""
    df = pd.read_excel(xlsx_path)
    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        cell = str(r["cells"]).strip()
        text = str(r["transcribed_text"]).strip()
        region = str(r["region"]).strip()
        if not cell or cell == "nan" or not text or text == "nan":
            continue
        rec = out.setdefault(cell, {"towns": Counter(), "entries": []})
        if region and region != "nan" and TOWN_RE.fullmatch(region):
            rec["towns"][region] += 1
        rec["entries"].append(parse_gt_line(text, r["phone_extension"]))
    return {
        cell: {"town": rec["towns"].most_common(1)[0][0] if rec["towns"] else "",
               "entries": rec["entries"]}
        for cell, rec in out.items() if rec["entries"]
    }


# ── Column-strip mode ────────────────────────────────────────────────────────
# Reproducing the ORIGINAL horizontal-rule segmentation turned out to be
# unreliable (OpenCV 5.x vs the 4.x used in the May 2026 digitisation run
# finds fewer rules → cell names shift → 23/25 smoke pages failed the
# drift guard). The 3 full-height VERTICAL rules, however, are robust, and
# the xlsx "cells" column encodes the column index in its name (colN_secSS),
# so joining gt to a COLUMN STRIP crop only needs: colN prefix → strip N,
# sections concatenated in secSS order. Targets are ~¼ page (≈4-5k tokens)
# — small enough for bf16 LoRA loss on a 40GB A100.

COLUMN_PROMPT = """\
You are an expert reader of the 1942 German Reichstelefonbuch (printed \
telephone directory, Antiqua typeface with bold surnames).

This image is ONE COLUMN of a directory page. It contains one or more town \
sections: a bold town header (possibly with the state in parentheses/\
brackets) followed by one subscriber per line: phone number, then name, \
then profession/business, sometimes a street.

Transcribe the whole column as JSON:

{
  "sections": [
    {
      "town": "<town header verbatim, e.g. 'Mattsee (Salzburg)'>",
      "entries": [
        {
          "extension": "<phone number as printed>",
          "surname": "<family name or institution name (first bold word(s))>",
          "given_name": "<given name(s) or initials if present>",
          "profession": "<occupation/business if present>",
          "raw": "<the complete line verbatim>"
        }
      ]
    }
  ]
}

Rules:
1. One section per town header, top to bottom. One entry per printed line.
2. If the column starts mid-section (no header at top), use "" as that \
section's town.
3. Omit a field entirely when the line does not give it (institutions like \
'Gendarmerie' have only a surname field).
4. Keep original spelling and abbreviations verbatim; do not modernise.
5. Do NOT guess. Wrap any smudged or ambiguous reading in ⟦double \
brackets⟧ inside the field value, e.g. "surname": "⟦L?gl⟧".
6. Output ONLY the JSON object — no Markdown fences, no commentary.
"""

CELL_NAME_RE = re.compile(r"col(\d+)_sec(\d+)\.png")


def segment_columns(img: np.ndarray) -> list[np.ndarray]:
    """Column strips between the full-height vertical rules (the robust
    half of cv_segment_test.py's detection)."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK, ADAPTIVE_C)
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, V_LINE_MIN_LEN))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kern, iterations=2)
    v_xs_raw = [x for x in range(W) if v_lines[:, x].sum() > 0]
    col_splits = cluster(v_xs_raw, CLUSTER_GAP)
    x_bounds = [0] + col_splits + [W]
    return [img[:, max(0, x_bounds[i] + CROP_MARGIN // 2):
                x_bounds[i + 1] - CROP_MARGIN // 2]
            for i in range(len(x_bounds) - 1)]


def alpha_fraction(entries: list[dict]) -> float:
    """Share of entries whose surname contains a letter — filters the
    number-only noise cells observed in the ground truth."""
    if not entries:
        return 0.0
    ok = sum(1 for e in entries
             if any(c.isalpha() for c in e.get("surname", "")))
    return ok / len(entries)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtb-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--volumes", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--max-pages-per-volume", type=int, default=None)
    ap.add_argument("--min-entries", type=int, default=3,
                    help="skip columns with fewer gold entries")
    ap.add_argument("--min-alpha-fraction", type=float, default=0.5,
                    help="skip columns where fewer than this share of gold "
                         "surnames contain a letter (number-only noise "
                         "cells in the master xlsx)")
    ap.add_argument("--max-edge", type=int, default=3200,
                    help="resize crops so the longest edge is at most this "
                         "many px. At 600 DPI a column strip is ~7500 px "
                         "tall ≈ 11,700 vision tokens, which alone OOMs the "
                         "fp32 loss upcast (seq × vocab 248,320). 3200 px "
                         "→ ~2,100 vision tokens and ~30 px line height, "
                         "still comfortably legible for print.")
    args = ap.parse_args()

    images_dir = args.out_dir / "images"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"

    n_written = n_cols_skipped = n_pages_skipped = 0
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for vol in args.volumes:
            name = VOLUMES[vol]
            pdf_path = args.rtb_dir / f"{name}.pdf"
            xlsx_dir = args.rtb_dir / "RTB_results_1_2_3_4" / f"results_{name}"
            xlsx_files = sorted(xlsx_dir.glob("page_*_ocr.xlsx"))
            if args.max_pages_per_volume:
                xlsx_files = xlsx_files[:args.max_pages_per_volume]
            print(f"vol {vol} ({name}): {len(xlsx_files)} pages",
                  file=sys.stderr)

            for xlsx_path in xlsx_files:
                page_num = int(re.search(r"page_(\d+)_ocr", xlsx_path.name)
                               .group(1))
                gt_cells = load_cells_ground_truth(xlsx_path)
                if not gt_cells:
                    n_pages_skipped += 1
                    continue

                # group gt cells into columns via the colN_secSS name
                columns: dict[int, list[tuple[int, dict]]] = {}
                bad_name = False
                for cell, gold in gt_cells.items():
                    m = CELL_NAME_RE.fullmatch(cell)
                    if not m:
                        bad_name = True
                        break
                    columns.setdefault(int(m.group(1)), []).append(
                        (int(m.group(2)), gold))
                if bad_name:
                    n_pages_skipped += 1
                    continue

                try:
                    img = render_page_600dpi(pdf_path, page_num)
                    strips = segment_columns(img)
                except Exception as e:
                    print(f"  vol{vol} p{page_num}: render/segment failed: "
                          f"{e}", file=sys.stderr)
                    n_pages_skipped += 1
                    continue

                # drift guard, column edition: every gt column index must
                # have a corresponding strip
                if max(columns) > len(strips):
                    print(f"  vol{vol} p{page_num}: gt has col{max(columns)} "
                          f"but only {len(strips)} strips detected — skip",
                          file=sys.stderr)
                    n_pages_skipped += 1
                    continue

                page_dir = images_dir / f"vol{vol}" / f"page_{page_num:04d}"
                for ci, secs in sorted(columns.items()):
                    secs.sort(key=lambda t: t[0])
                    target = {"sections": [g for _, g in secs]}
                    entries = [e for _, g in secs for e in g["entries"]]
                    if (len(entries) < args.min_entries
                            or alpha_fraction(entries) < args.min_alpha_fraction):
                        n_cols_skipped += 1
                        continue
                    page_dir.mkdir(parents=True, exist_ok=True)
                    crop_path = page_dir / f"col{ci}.png"
                    if not crop_path.exists():
                        strip = strips[ci - 1]
                        h, w = strip.shape[:2]
                        scale = args.max_edge / max(h, w)
                        if scale < 1.0:
                            strip = cv2.resize(
                                strip, (int(w * scale), int(h * scale)),
                                interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(crop_path), strip)
                    manifest.write(json.dumps({
                        "image": str(crop_path),
                        "volume": vol, "page": page_num, "column": ci,
                        "prompt": COLUMN_PROMPT,
                        "target": json.dumps(target, ensure_ascii=False),
                        "n_entries": len(entries),
                    }, ensure_ascii=False) + "\n")
                    n_written += 1

    print(f"\nWrote {n_written} column examples to {manifest_path} "
          f"(skipped: {n_cols_skipped} columns, {n_pages_skipped} pages)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
