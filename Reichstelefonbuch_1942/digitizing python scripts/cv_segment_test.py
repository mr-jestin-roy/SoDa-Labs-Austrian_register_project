"""
cv_segment_test.py
──────────────────
Segmentation for dense multi-column historical German telephone directories.

Strategy
────────
1. Detect the 3 full-height vertical ruling lines  → split page into 4 columns.
2. For each column strip independently, run horizontal line detection with a
   shorter kernel (fraction of column width).  Horizontal rules don't span all
   columns, so per-column detection is the only correct approach.
3. Crop each (column, section) cell and save for OCR.

Run:
    venv_layoutparser/bin/python cv_segment_test.py

Outputs:
    cv_segment_lines.png     — vertical rules + per-column horizontal rules
    cv_segment_cols.png      — all detected cells drawn on the original
    cv_segment_cells/        — individual crops named col<C>_sec<S>.png
"""

import os, sys
import cv2
import numpy as np
from PIL import Image


# Read source image DPI once so outputs inherit it (cv2 strips DPI metadata)
def _source_dpi(path, default=(600, 600)):
    try:
        with Image.open(path) as im:
            return im.info.get("dpi", default)
    except Exception:
        return default


def imwrite_with_dpi(path, bgr_img, dpi):
    """cv2.imwrite strips DPI metadata → re-save via PIL to embed it."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path, dpi=dpi)


# ── Paths (override via env vars for orchestration) ──────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Reichstelefonbuch_1942/
INPUT     = os.getenv("INPUT_IMAGE", os.path.join(BASE, "page_749-0749.png"))
OUT_LINES = os.getenv("OUT_LINES",   os.path.join(BASE, "cv_segment_lines.png"))
OUT_COLS  = os.getenv("OUT_COLS",    os.path.join(BASE, "cv_segment_cols.png"))
OUT_CELLS = os.getenv("OUT_CELLS",   os.path.join(BASE, "cv_segment_cells"))

print("=" * 60)
print("▶ ENTERING: cv_segment_test.py")
print("=" * 60)
print(f"  INPUT:     {INPUT}")
print(f"  OUT_LINES: {OUT_LINES}")
print(f"  OUT_COLS:  {OUT_COLS}")
print(f"  OUT_CELLS: {OUT_CELLS}")
print("-" * 60)

# ── Tuning ────────────────────────────────────────────────────────────────────
ADAPTIVE_BLOCK = 21  # adaptive threshold block size (odd)
ADAPTIVE_C = 10  # subtracted from mean
V_LINE_MIN_LEN = 600  # px — vertical rule must span this height (600 DPI)
H_FRAC_OF_COL = 0.05  # horizontal rule must span this fraction of column width
CLUSTER_GAP = 45  # px — merge line positions closer than this
CROP_MARGIN = 18  # px — inset from detected border when cropping

# ── 1. Load & threshold ───────────────────────────────────────────────────────
img = cv2.imread(INPUT)
if img is None:
    sys.exit(f"Cannot read: {INPUT}")

DPI = _source_dpi(INPUT)
H, W = img.shape[:2]

LINE_THICK = 3    # divider lines
BOX_THICK  = 3    # cell rectangles
FONT_SCALE = 1.65
FONT_THICK = 3
DPI_SCALE  = 3    # kept only for the text y-offset calculation

print(f"Image: {W}×{H} px  DPI: {DPI}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    ADAPTIVE_BLOCK,
    ADAPTIVE_C,
)

# ── 2. Detect vertical column dividers (full-height) ─────────────────────────
v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, V_LINE_MIN_LEN))
v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kern, iterations=2)


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


v_xs_raw = [x for x in range(W) if v_lines[:, x].sum() > 0]
col_splits = cluster(v_xs_raw, CLUSTER_GAP)
print(f"Vertical dividers at x : {col_splits}")

# Column x-boundaries: page edges + divider positions
x_bounds = [0] + col_splits + [W]
n_cols = len(x_bounds) - 1

# ── 3. Per-column horizontal rule detection ───────────────────────────────────
vis_lines = img.copy()

# Draw thick green lines at clustered x-positions
for x in col_splits:
    cv2.line(vis_lines, (x, 0), (x, H), (0, 220, 0), LINE_THICK)

all_cells = []  # list of (x1, y1, x2, y2, col_idx, sec_idx)

for ci in range(n_cols):
    cx1, cx2 = x_bounds[ci], x_bounds[ci + 1]
    col_w = cx2 - cx1
    h_min_len = max(30, int(col_w * H_FRAC_OF_COL))

    # Run horizontal detection on this column's strip only
    col_thresh = thresh[:, cx1:cx2]
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (h_min_len, 1))
    h_lines_col = cv2.morphologyEx(col_thresh, cv2.MORPH_OPEN, h_kern, iterations=2)

    # Cluster the detected y-positions for this column
    h_ys_raw = [y for y in range(H) if h_lines_col[y, :].sum() > 0]
    row_splits = cluster(h_ys_raw, CLUSTER_GAP)

    # Draw thick red lines at clustered y-positions (within this column's x-range)
    for y in row_splits:
        cv2.line(vis_lines, (cx1, y), (cx2, y), (0, 0, 255), LINE_THICK)
    print(
        f"  Col {ci+1} (x={cx1}–{cx2}, h_min={h_min_len}px) "
        f"→ horizontal rules at y={row_splits}"
    )

    # Section y-boundaries: page edges + per-column rule positions
    y_bounds = [0] + row_splits + [H]

    for si in range(len(y_bounds) - 1):
        y1 = y_bounds[si] + CROP_MARGIN
        y2 = y_bounds[si + 1] - CROP_MARGIN
        x1 = cx1 + CROP_MARGIN
        x2 = cx2 - CROP_MARGIN
        if (x2 - x1) > 20 and (y2 - y1) > 20:
            all_cells.append((x1, y1, x2, y2, ci + 1, si + 1))

imwrite_with_dpi(OUT_LINES, vis_lines, DPI)
print(f"\nLine overlay saved  → {OUT_LINES}")
print(f"Total cells: {len(all_cells)}")

# ── 4. Draw all cells on the original ────────────────────────────────────────
PALETTE = [
    (0, 180, 0),
    (220, 0, 220),
    (0, 160, 255),
    (255, 80, 0),
    (0, 200, 200),
    (180, 0, 0),
    (0, 80, 255),
]

cols_vis = img.copy()
for x1, y1, x2, y2, ci, si in all_cells:
    color = PALETTE[(ci - 1) % len(PALETTE)]
    cv2.rectangle(cols_vis, (x1, y1), (x2, y2), color, BOX_THICK)
    cv2.putText(
        cols_vis,
        f"c{ci}s{si}",
        (x1 + 4, y1 + 22 * DPI_SCALE),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        color,
        FONT_THICK,
        cv2.LINE_AA,
    )

imwrite_with_dpi(OUT_COLS, cols_vis, DPI)
print(f"Cell boxes saved    → {OUT_COLS}")

# ── 5. Crop and save cells ────────────────────────────────────────────────────
os.makedirs(OUT_CELLS, exist_ok=True)
for x1, y1, x2, y2, ci, si in all_cells:
    crop = img[y1:y2, x1:x2]
    fname = f"col{ci}_sec{si:02d}.png"
    imwrite_with_dpi(os.path.join(OUT_CELLS, fname), crop, DPI)

print(f"Crops saved         → {OUT_CELLS}/  ({len(all_cells)} files)")
print("=" * 60)
print("✔ EXITING: cv_segment_test.py (SUCCESS)")
print("=" * 60)
