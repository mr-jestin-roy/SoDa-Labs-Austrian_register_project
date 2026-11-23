"""
LayoutParser layout detection test on a Pernegg Parish document page.

Install (run once — use the conda env's own pip, not system pip):
    conda activate qwen_ocr
    ~/miniconda3/envs/qwen_ocr/bin/pip install layoutparser torchvision
    ~/miniconda3/envs/qwen_ocr/bin/pip install "git+https://github.com/facebookresearch/detectron2.git@v0.5#egg=detectron2"
    ~/miniconda3/envs/qwen_ocr/bin/pip install "layoutparser[layoutmodels]"

Note: layoutparser is pip-only — it does not exist on conda-forge.

Run:
    python layoutparser_test.py
"""

import sys
import os

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "page_749-0749.png")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "layoutparser_output.png")

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    sys.exit("Missing: opencv-python  →  pip install opencv-python")

try:
    import layoutparser as lp
except ImportError:
    sys.exit(
        "Missing: layoutparser  (pip-only, not on conda-forge)\n"
        "  ~/miniconda3/envs/qwen_ocr/bin/pip install layoutparser torchvision\n"
        '  ~/miniconda3/envs/qwen_ocr/bin/pip install "layoutparser[layoutmodels]"'
    )

# detectron2 must be installed separately — it is NOT bundled with layoutparser
try:
    import detectron2  # noqa: F401
except ImportError:
    sys.exit(
        "Missing: detectron2  (must be built from source)\n"
        "  Note: use the LATEST main branch — v0.5 does not support torch 2.x\n"
        '  ~/miniconda3/envs/qwen_ocr/bin/pip install "git+https://github.com/facebookresearch/detectron2.git"\n'
        "  (takes ~5-10 min to compile — run once and wait)"
    )

# Verify the Detectron2LayoutModel is reachable after detectron2 is present
if not hasattr(lp, "Detectron2LayoutModel"):
    sys.exit(
        "layoutparser loaded but Detectron2LayoutModel is unavailable.\n"
        "Re-install layoutparser[layoutmodels] after detectron2 is in place:\n"
        '  ~/miniconda3/envs/qwen_ocr/bin/pip install --force-reinstall "layoutparser[layoutmodels]"'
    )

# ── Load image ────────────────────────────────────────────────────────────────
if not os.path.exists(IMAGE_PATH):
    sys.exit(f"Image not found: {IMAGE_PATH}")

image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    sys.exit(f"cv2 could not read: {IMAGE_PATH}")

image_rgb = image_bgr[..., ::-1]   # BGR → RGB
print(f"Image loaded: {image_rgb.shape[1]}×{image_rgb.shape[0]} px")

# ── Model ─────────────────────────────────────────────────────────────────────
# PubLayNet — good for printed-document layouts (text, title, list, table, figure)
# Prima      — better for historical/newspaper layouts; swap model path to use it
#
# PubLayNet label map:  0=Text  1=Title  2=List  3=Table  4=Figure
# Prima      label map: 0=TextRegion  1=ImageRegion  2=LineDrawingRegion
#                        3=GraphicRegion  4=TabularRegion  5=ChartRegion
#                        6=SeparatorRegion  7=MathsRegion  8=NoiseRegion
#                        9=FrameRegion  10=CaptionRegion  11=AdvertRegion  12=OtherRegion

LABEL_MAP = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

SCORE_THRESHOLD = 0.5   # lower → more detections, potentially noisier

# ── Download model files from HuggingFace if not already cached ───────────────
# The original lp:// URLs pointed to Dropbox, which has since deleted the files.
# We download directly and pass local paths to avoid iopath's broken HTTP handler.
import urllib.request

_HF_BASE = "https://huggingface.co/layoutparser/detectron2/resolve/main/PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x"
_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "layoutparser", "PubLayNet_X101")
os.makedirs(_CACHE_DIR, exist_ok=True)

_CONFIG_PATH = os.path.join(_CACHE_DIR, "config.yml")
_MODEL_PATH  = os.path.join(_CACHE_DIR, "model_final.pth")

def _download(url, dest):
    print(f"  Downloading {os.path.basename(dest)} …")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved → {dest}")

if not os.path.exists(_CONFIG_PATH):
    _download(f"{_HF_BASE}/config.yml", _CONFIG_PATH)
else:
    print(f"Config cached: {_CONFIG_PATH}")

if not os.path.exists(_MODEL_PATH):
    print("Downloading model weights (856 MB) — this will take a few minutes …")
    _download(f"{_HF_BASE}/model_final.pth", _MODEL_PATH)
else:
    print(f"Model cached:  {_MODEL_PATH}")

print(f"\nLoading model …\n")

model = lp.Detectron2LayoutModel(
    _CONFIG_PATH,
    model_path=_MODEL_PATH,
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", SCORE_THRESHOLD],
    label_map=LABEL_MAP,
)

# ── Detect ────────────────────────────────────────────────────────────────────
print("Running layout detection …")
layout = model.detect(image_rgb)

# ── Report ────────────────────────────────────────────────────────────────────
print(f"\nDetected {len(layout)} regions:\n")
for i, block in enumerate(layout):
    x1, y1, x2, y2 = (
        int(block.block.x_1), int(block.block.y_1),
        int(block.block.x_2), int(block.block.y_2),
    )
    w, h = x2 - x1, y2 - y1
    print(
        f"  [{i+1:02d}] {block.type:<10}  "
        f"score={block.score:.3f}  "
        f"bbox=({x1},{y1})→({x2},{y2})  {w}×{h}px"
    )

# ── Visualise & save ──────────────────────────────────────────────────────────
# Colour scheme per region type
COLORS = {
    "Text":   (0,   128, 255),
    "Title":  (255, 0,   0  ),
    "List":   (0,   200, 0  ),
    "Table":  (200, 0,   200),
    "Figure": (255, 165, 0  ),
}

vis = image_bgr.copy()
for block in layout:
    x1, y1, x2, y2 = (
        int(block.block.x_1), int(block.block.y_1),
        int(block.block.x_2), int(block.block.y_2),
    )
    color = COLORS.get(block.type, (128, 128, 128))
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
    label = f"{block.type} {block.score:.2f}"
    cv2.putText(
        vis, label, (x1, max(y1 - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
    )

cv2.imwrite(OUTPUT_PATH, vis)
print(f"\nAnnotated image saved → {OUTPUT_PATH}")

# ── Optional: also try LayoutParser's built-in visualiser ────────────────────
try:
    viz = lp.draw_box(image_rgb, layout, box_width=3)
    lp_output = OUTPUT_PATH.replace(".png", "_lp_native.png")
    viz.save(lp_output)
    print(f"LayoutParser native viz saved → {lp_output}")
except Exception as e:
    print(f"(LayoutParser native viz skipped: {e})")
