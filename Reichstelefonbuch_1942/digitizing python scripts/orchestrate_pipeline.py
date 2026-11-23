"""
orchestrate_pipeline.py
───────────────────────
Master script that processes a range of pages from a PDF through the full pipeline:

    PDF page → 600 DPI PNG → cv_segment → ocr_cells_chandra → json_to_xlsx

All outputs are named with the page identifier for full traceability.

Usage:
    python orchestrate_pipeline.py input.pdf --start 10 --end 20
    python orchestrate_pipeline.py input.pdf --pages 10,15,20,25
    python orchestrate_pipeline.py input.pdf --start 1 --end 100 --workers 8
    # Log to a file (stdout + stderr) and still see the terminal:
    python orchestrate_pipeline.py book.pdf --start 209 --end 1347 2>&1 | tee results/run.log

Outputs per page (e.g. page 749):
    results/page_749_ocr.json                    — OCR results (kept)
    results/page_749_ocr.xlsx                    — final spreadsheet (kept)
    (After a successful run, the large intermediates below are removed by default:)
    pages/page_749.png                          — 600 DPI PNG from PDF
    segments/page_749/cv_segment_lines.png      — line overlay
    segments/page_749/cv_segment_cols.png        — cell boxes overlay
    segments/page_749/cells/*.png                — cell crops and grid_*.png
    Use --keep-intermediates to retain the above for debugging.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Missing: PyMuPDF  →  python -m pip install pymupdf")

# ── Config ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent   # Reichstelefonbuch_1942/
SCRIPTS_DIR = Path(__file__).parent          # digitizing python scripts/
PAGES_DIR   = BASE / "pages"
SEGMENTS_DIR = BASE / "segments"
RESULTS_DIR  = BASE / "results"

VLLM_URL    = os.getenv("VLLM_URL", "http://localhost:8010/v1/chat/completions")
DPI         = 600
# Parallel pages (vLLM --max-num-seqs is often 8; align workers to avoid overload)
WORKERS     = 8

# ── PDF to PNG ────────────────────────────────────────────────────────────────
def pdf_page_to_png(pdf_path: Path, page_num: int, output_path: Path) -> bool:
    """Extract a single page from PDF at 600 DPI and save as PNG."""
    doc = fitz.open(pdf_path)
    if page_num < 1 or page_num > len(doc):
        print(f"  ✗ Page {page_num} out of range (PDF has {len(doc)} pages)")
        doc.close()
        return False

    page = doc[page_num - 1]  # 0-indexed
    zoom = DPI / 72  # PDF default is 72 DPI
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(output_path))
    doc.close()

    print(f"  ✓ Extracted page {page_num} → {output_path.name} ({pix.width}×{pix.height} px, {DPI} DPI)")
    return True


# ── Run child scripts ─────────────────────────────────────────────────────────
def run_script(script_name: str, env_vars: dict) -> bool:
    """Run a script with environment variables, return True if successful."""
    script_path = SCRIPTS_DIR / script_name
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_vars.items()})

    result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  ✗ {script_name} failed:")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return False

    # Print key lines from stdout
    for line in result.stdout.strip().split("\n"):
        if line.strip():
            print(f"    {line}")
    return True


def cleanup_intermediates(png_path: Path, segment_dir: Path) -> None:
    """Delete page PNG and entire segments/<page_id>/ (cells, line/cols overlays, grids)."""
    try:
        if png_path.is_file():
            png_path.unlink()
    except OSError as e:
        print(f"  ⚠ could not remove {png_path}: {e}")
    try:
        if segment_dir.is_dir():
            shutil.rmtree(segment_dir)
    except OSError as e:
        print(f"  ⚠ could not remove {segment_dir}: {e}")


# ── Process single page ───────────────────────────────────────────────────────
def process_page(
    pdf_path: Path, page_num: int, *, keep_intermediates: bool = False
) -> dict:
    """Run full pipeline for a single page. Returns timing dict or None on failure."""
    page_id = f"page_{page_num:04d}"
    print(f"\n{'='*60}")
    print(f"Processing {page_id}")
    print(f"{'='*60}")

    timing = {"pdf_extract": 0, "segment": 0, "ocr": 0, "xlsx": 0, "total": 0}
    page_start = time.time()

    # Paths for this page
    png_path       = PAGES_DIR / f"{page_id}.png"
    segment_dir    = SEGMENTS_DIR / page_id
    cells_dir      = segment_dir / "cells"
    lines_out      = segment_dir / "cv_segment_lines.png"
    cols_out       = segment_dir / "cv_segment_cols.png"
    json_out       = RESULTS_DIR / f"{page_id}_ocr.json"
    xlsx_out       = RESULTS_DIR / f"{page_id}_ocr.xlsx"

    # Step 1: PDF → PNG
    print("\n[1/4] Extracting page from PDF...")
    t0 = time.time()
    if not pdf_page_to_png(pdf_path, page_num, png_path):
        return None
    timing["pdf_extract"] = time.time() - t0
    print(f"  ⏱ {timing['pdf_extract']:.1f}s")

    # Step 2: Segmentation
    print("\n[2/4] Running cv_segment_test.py...")
    t0 = time.time()
    segment_dir.mkdir(parents=True, exist_ok=True)
    cells_dir.mkdir(parents=True, exist_ok=True)

    success = run_script("cv_segment_test.py", {
        "INPUT_IMAGE": str(png_path),
        "OUT_LINES": str(lines_out),
        "OUT_COLS": str(cols_out),
        "OUT_CELLS": str(cells_dir),
    })
    if not success:
        return None
    timing["segment"] = time.time() - t0
    print(f"  ⏱ {timing['segment']:.1f}s")

    # Step 3: OCR with Chandra
    print("\n[3/4] Running ocr_cells_chandra.py...")
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    success = run_script("ocr_cells_chandra.py", {
        "CELLS_DIR": str(cells_dir),
        "GRIDS_DIR": str(cells_dir),
        "OUT_JSON": str(json_out),
        "VLLM_URL": VLLM_URL,
        "PDF_FILENAME": pdf_path.name,
        "FAILED_CSV": str(RESULTS_DIR / "failed_cells.csv"),
    })
    if not success:
        return None
    timing["ocr"] = time.time() - t0
    print(f"  ⏱ {timing['ocr']:.1f}s")

    # Step 4: JSON → XLSX
    print("\n[4/4] Running json_to_xlsx.py...")
    t0 = time.time()
    success = run_script("json_to_xlsx.py", {
        "IN_JSON": str(json_out),
        "OUT_XLSX": str(xlsx_out),
    })
    if not success:
        return None
    timing["xlsx"] = time.time() - t0
    print(f"  ⏱ {timing['xlsx']:.1f}s")

    if not keep_intermediates:
        cleanup_intermediates(png_path, segment_dir)
        print(f"  🗑 removed intermediates: {png_path.name}, {segment_dir.relative_to(BASE)}/")

    timing["total"] = time.time() - page_start
    print(f"\n✓ {page_id} complete → {xlsx_out}")
    print(f"  ⏱ Total: {timing['total']:.1f}s")
    return timing


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process PDF pages through the full OCR pipeline"
    )
    parser.add_argument("pdf", type=Path, help="Input PDF file")
    parser.add_argument("--start", type=int, help="Start page number (1-indexed)")
    parser.add_argument("--end", type=int, help="End page number (inclusive)")
    parser.add_argument("--pages", type=str, help="Comma-separated list of page numbers")
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS,
        help=f"How many pages to run in parallel (default: {WORKERS})",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep pages/ PNG and segments/ (lines, cols, cell crops, grids) after success",
    )

    args = parser.parse_args()
    if not args.pdf.exists():
        sys.exit(f"PDF not found: {args.pdf}")

    # Determine which pages to process
    doc = fitz.open(args.pdf)
    total_pages = len(doc)
    doc.close()
    print(f"PDF: {args.pdf.name} ({total_pages} pages)")

    if args.pages:
        page_nums = [int(p.strip()) for p in args.pages.split(",")]
    elif args.start and args.end:
        page_nums = list(range(args.start, args.end + 1))
    elif args.start:
        page_nums = list(range(args.start, total_pages + 1))
    elif args.end:
        page_nums = list(range(1, args.end + 1))
    else:
        sys.exit("Specify --start/--end or --pages")

    # Filter valid pages
    page_nums = [p for p in page_nums if 1 <= p <= total_pages]
    print(f"Pages to process: {page_nums}")
    n_workers = max(1, min(args.workers, len(page_nums)))
    if n_workers > 1:
        print(f"Parallel workers: {n_workers}")

    # Create output directories
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Process pages in parallel (log lines from different pages may interleave)
    results = {}
    timings = {}
    run_start = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        timings_list = list(
            executor.map(
                partial(
                    process_page,
                    args.pdf,
                    keep_intermediates=args.keep_intermediates,
                ),
                page_nums,
            )
        )

    for page_num, timing in zip(page_nums, timings_list):
        if timing:
            results[page_num] = "✓"
            timings[page_num] = timing
        else:
            results[page_num] = "✗"

    run_total = time.time() - run_start

    # Summary
    print(f"\n{'='*60}")
    print("TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Page':<12} {'PDF→PNG':>8} {'Segment':>8} {'OCR':>8} {'XLSX':>8} {'Total':>8}")
    print("-" * 60)

    for page_num, status in results.items():
        if page_num in timings:
            t = timings[page_num]
            print(f"page_{page_num:04d}   {t['pdf_extract']:>7.1f}s {t['segment']:>7.1f}s {t['ocr']:>7.1f}s {t['xlsx']:>7.1f}s {t['total']:>7.1f}s")
        else:
            print(f"page_{page_num:04d}   {'FAILED':>44}")

    print("-" * 60)

    # Averages for estimation
    if timings:
        avg_total = sum(t["total"] for t in timings.values()) / len(timings)
        avg_pdf   = sum(t["pdf_extract"] for t in timings.values()) / len(timings)
        avg_seg   = sum(t["segment"] for t in timings.values()) / len(timings)
        avg_ocr   = sum(t["ocr"] for t in timings.values()) / len(timings)
        avg_xlsx  = sum(t["xlsx"] for t in timings.values()) / len(timings)

        print(f"{'AVERAGE':<12} {avg_pdf:>7.1f}s {avg_seg:>7.1f}s {avg_ocr:>7.1f}s {avg_xlsx:>7.1f}s {avg_total:>7.1f}s")
        print()

        # Estimation helper
        print(f"📊 Estimation formulas (based on {len(timings)} page(s)):")
        print(f"   Per page     : ~{avg_total:.0f}s ({avg_total/60:.1f} min)")
        print(f"   10 pages     : ~{avg_total*10/60:.1f} min")
        print(f"   100 pages    : ~{avg_total*100/60:.1f} min ({avg_total*100/3600:.1f} hr)")
        print(f"   {total_pages} pages (full PDF): ~{avg_total*total_pages/60:.1f} min ({avg_total*total_pages/3600:.1f} hr)")

    print()
    succeeded = sum(1 for s in results.values() if s == "✓")
    print(f"✓ {succeeded}/{len(results)} pages processed successfully")
    print(f"⏱ Total run time: {run_total:.1f}s ({run_total/60:.1f} min)")


if __name__ == "__main__":
    main()
