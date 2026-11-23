"""
ocr_cells_chandra.py
────────────────────
Takes cropped segment cells from cv_segment_cells/, tiles them into a 3×3
grid (or 2×2 if cells are large) with 10 px padding, then sends each
composite image to a ChandraOCR vLLM endpoint for structured extraction of
the 1942 German telephone directory (Reichstelefonbuch).

Extracted per entry:
  - region_name  — bold town/region heading at the top of each cell
  - entries      — list of {extension, text} pairs

Run:
    venv_layoutparser/bin/python ocr_cells_chandra.py

    # Set the vLLM endpoint (pick one):
    VLLM_URL=http://my-server:8010/v1/chat/completions \
        venv_layoutparser/bin/python ocr_cells_chandra.py
    VLM_BASE_URL=http://my-server:8010 venv_layoutparser/bin/python ocr_cells_chandra.py

Output:
    ocr_results_chandra.json        — nested JSON with all extracted data
    cv_segment_cells/grid_NN.png    — composite grid images sent to Chandra
                                      (saved alongside the individual cell crops)
"""

import os, sys, json, base64, math, textwrap, time
from pathlib import Path

import cv2
import numpy as np
import requests

# ── Config ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
# orchestrate_pipeline.py sets CELLS_DIR / GRIDS_DIR / OUT_JSON per page
CELLS_DIR = Path(os.environ.get("CELLS_DIR", str(BASE / "cv_segment_cells")))
GRIDS_DIR = Path(os.environ.get("GRIDS_DIR", str(CELLS_DIR)))
OUT_JSON  = Path(os.environ.get("OUT_JSON", str(BASE / "ocr_results_chandra.json")))


def _resolve_vllm_chat_url() -> str:
    """OpenAI-style chat/completions URL. VLLM_URL wins; else VLM_BASE_URL + /v1/chat/completions."""
    u = os.getenv("VLLM_URL")
    if u is not None and u.strip():
        return u.strip()
    base = (os.getenv("VLM_BASE_URL") or "").strip().rstrip("/")
    if base:
        if "/v1/" in base:
            return base
        return f"{base}/v1/chat/completions"
    return "http://localhost:8010/v1/chat/completions"


VLLM_URL   = _resolve_vllm_chat_url()
MODEL      = "chandra"

MAX_TILE_W = 560    # px — each cell is scaled to fit within this width
MAX_TILE_H = 900    # px — and this height (aspect ratio preserved)
PADDING    = 10     # px — gap between tiles and around the border
BATCH_SIZE = 9      # max cells per grid (fills 3×3; drops to 2×2 if tiles are large)

GRID_3x3_MAX_AREA = 3800 * 3800   # if composite exceeds this, use 2×2 grid instead

PROMPT = textwrap.dedent("""\
    This image contains one or more cells from a 1942 German telephone directory
    (Reichstelefonbuch). Each cell covers a specific town or region in Austria/Germany.

    For EVERY cell visible, extract the following as structured data:

    1. region_name  — the bold town/region heading at the top of the cell
                      (e.g. "Mattsee", "Maulbronn", "Mauerbach")
    2. entries      — a list of telephone entries, each with:
         - extension : the number that appears before the name/text
                       (these are the local telephone extension numbers)
         - text      : the full name, business, or description that follows
                       the extension number on the same line

    Rules:
    - If a line has no extension number, it is a continuation of the previous
      entry or a sub-heading — include it in the text of the nearest entry above.
    - Preserve German characters (ä, ö, ü, ß).
    - If a cell contains an advertisement or non-entry block, set entries to []
      and describe it briefly in region_name.

    Return ONLY valid JSON — no markdown fences, no extra commentary:
    [
      {
        "region_name": "...",
        "entries": [
          {"extension": "12", "text": "Müller, Hans, Bäckerei"},
          {"extension": "34", "text": "Gasthof zum Adler"}
        ]
      }
    ]
""")

# ── Image helpers ─────────────────────────────────────────────────────────────
def scale_to_tile(img, max_w, max_h):
    """Scale image to fit within (max_w × max_h), preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def make_tile(img, tile_w, tile_h):
    """Paste img onto a white tile of exactly (tile_w × tile_h)."""
    tile = np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 255
    h, w = img.shape[:2]
    tile[:h, :w] = img
    return tile


def choose_cols(images):
    """Return 3 (3×3 grid) unless the composite would be very large → 2 (2×2)."""
    scaled = [scale_to_tile(im, MAX_TILE_W, MAX_TILE_H) for im in images]
    tile_w = max(im.shape[1] for im in scaled)
    tile_h = max(im.shape[0] for im in scaled)
    cols3_area = (3 * tile_w + 4 * PADDING) * (3 * tile_h + 4 * PADDING)
    return 2 if cols3_area > GRID_3x3_MAX_AREA else 3


def build_grid(images, cols, padding):
    """Arrange BGR images into a cols-column grid with uniform tile size."""
    scaled = [scale_to_tile(im, MAX_TILE_W, MAX_TILE_H) for im in images]
    tile_w = max(im.shape[1] for im in scaled)
    tile_h = max(im.shape[0] for im in scaled)
    tiles  = [make_tile(im, tile_w, tile_h) for im in scaled]

    rows = math.ceil(len(tiles) / cols)
    while len(tiles) < rows * cols:
        tiles.append(np.ones((tile_h, tile_w, 3), dtype=np.uint8) * 255)

    comp_w = cols * tile_w + (cols + 1) * padding
    comp_h = rows * tile_h + (rows + 1) * padding
    canvas = np.ones((comp_h, comp_w, 3), dtype=np.uint8) * 240  # light-grey bg

    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        y = padding + r * (tile_h + padding)
        x = padding + c * (tile_w + padding)
        canvas[y:y + tile_h, x:x + tile_w] = tile

    return canvas


def img_to_b64(img):
    """Encode a BGR numpy array as a base64 PNG string."""
    _, buf = cv2.imencode(".png", img)
    return base64.standard_b64encode(buf.tobytes()).decode()


# ── vLLM / Chandra call ───────────────────────────────────────────────────────
def call_chandra(b64_image: str, batch_index: int) -> str:
    """
    Send a base64-encoded PNG to the Chandra vLLM endpoint.
    Returns the raw text response.
    """
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    # Text prompt first — better prefix-cache hit rate
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 8192,
        "temperature": 0.1,
    }

    start = time.time()
    resp = requests.post(VLLM_URL, json=payload, timeout=600)
    resp.raise_for_status()
    elapsed = time.time() - start

    content = resp.json()["choices"][0]["message"]["content"]
    print(f"  Chandra response: {len(content)} chars  ({elapsed:.1f}s)")
    return content


def parse_json(raw: str, batch_idx: int, cell_names: list) -> object:
    """Strip accidental markdown fences and parse JSON; fall back to raw on error."""
    clean = (raw
             .removeprefix("```json")
             .removeprefix("```")
             .removesuffix("```")
             .strip())
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        print(f"  WARNING: JSON parse failed for batch {batch_idx+1} ({e}) — storing raw")
        return {"raw": raw, "cells": cell_names}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not VLLM_URL.startswith(("http://", "https://")):
        sys.exit(
            "Chandra vLLM URL is not configured.\n"
            "  export VLLM_URL=http://127.0.0.1:8010/v1/chat/completions\n"
            "  or export VLM_BASE_URL=http://127.0.0.1:8010"
        )

    # ── Load cell images ──────────────────────────────────────────────────────
    cell_files = sorted(CELLS_DIR.glob("*.png"))
    if not cell_files:
        sys.exit(f"No cell images found in {CELLS_DIR}\n"
                 "Run cv_segment_test.py first to generate the cell crops.")

    print(f"Found {len(cell_files)} cells in {CELLS_DIR}")

    cells = []
    for f in cell_files:
        img = cv2.imread(str(f))
        if img is not None:
            cells.append((f.name, img))
        else:
            print(f"  Warning: could not read {f.name}")

    # ── Group into batches of up to BATCH_SIZE ────────────────────────────────
    batches = [cells[i:i + BATCH_SIZE] for i in range(0, len(cells), BATCH_SIZE)]
    print(f"Grouped into {len(batches)} batch(es) → vLLM at {VLLM_URL}\n")

    GRIDS_DIR.mkdir(exist_ok=True)   # already exists (cell crops are here too)
    all_results = []

    for bi, batch in enumerate(batches):
        names  = [n for n, _ in batch]
        images = [im for _, im in batch]

        cols     = choose_cols(images)
        grid_img = build_grid(images, cols, PADDING)

        grid_path = GRIDS_DIR / f"grid_{bi+1:02d}.png"
        cv2.imwrite(str(grid_path), grid_img)

        h, w = grid_img.shape[:2]
        print(f"Batch {bi+1}/{len(batches)}  "
              f"cells={names}  grid={w}×{h}  layout={cols}×{math.ceil(len(batch)/cols)}")
        print(f"  Grid saved → {grid_path}")

        b64      = img_to_b64(grid_img)
        raw      = call_chandra(b64, bi)
        parsed   = parse_json(raw, bi, names)

        region_count = len(parsed) if isinstance(parsed, list) else "?"
        print(f"  Extracted {region_count} region(s)\n")

        all_results.append({
            "batch":       bi + 1,
            "cells":       names,
            "grid_layout": f"{cols}x{math.ceil(len(batch)/cols)}",
            "grid_image":  grid_path.name,
            "data":        parsed,
        })

    # ── Save JSON ─────────────────────────────────────────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_regions = sum(
        len(r["data"]) if isinstance(r["data"], list) else 0
        for r in all_results
    )
    print(f"Results saved  → {OUT_JSON}")
    print(f"Grid images    → {GRIDS_DIR}/")
    print(f"Total regions  : {total_regions}")


if __name__ == "__main__":
    main()
