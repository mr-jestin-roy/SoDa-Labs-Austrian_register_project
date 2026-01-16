import os
from PIL import Image
from pathlib import Path

# The 3 files that were hallucinating
HEAVY_FILES = ["Seite 63-64.jpg", "Seite 81-82.jpg", "Seite 83-84.jpg"]
INPUT_DIR = "./oesterreich_graz-seckau_Pernegg_8758"
OUTPUT_DIR = "./high_res_lr_splits"

def split_lr_lossless(filename):
    path = Path(INPUT_DIR) / filename
    if not path.exists(): return
    
    img = Image.open(path).convert("RGB")
    width, height = img.size
    mid = width // 2
    
    # Left and Right Crops
    left = img.crop((0, 0, mid, height))
    right = img.crop((mid, 0, width, height))
    
    base = path.stem
    # Use quality=100 and subsampling=0 to ensure maximum OCR sharpness
    left.save(Path(OUTPUT_DIR) / f"{base}_L.jpg", "JPEG", quality=100, subsampling=0)
    right.save(Path(OUTPUT_DIR) / f"{base}_R.jpg", "JPEG", quality=100, subsampling=0)
    print(f"✅ Lossless Split: {base} (L and R)")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
for f in HEAVY_FILES:
    split_lr_lossless(f)