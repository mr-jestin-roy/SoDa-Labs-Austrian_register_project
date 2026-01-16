import os
import torch
from PIL import Image
from pathlib import Path
from surya.detection import DetectionPredictor

# --- CONFIGURATION ---
INPUT_DIR = "./oesterreich_graz-seckau_Pernegg_8758/Split_and_process_table_images"
OUTPUT_DIR = "./smart_split_input"

# The 8 specific heavy files
HEAVY_FILES = [
    "Seite 19-20.jpg", "Seite 43-44.jpg", "Seite 53-54.jpg", 
    "Seite 63-64.jpg", "Seite 73-74.jpg", "Seite 81-82.jpg", 
    "Seite 83-84.jpg", "Seite 93-94.jpg"
]

def smart_split(filename, predictor):
    filepath = Path(INPUT_DIR) / filename
    if not filepath.exists():
        print(f"⚠️ Missing: {filename}")
        return

    print(f"🔍 Analyzing layout for {filename}...")
    try:
        # Open image
        image = Image.open(filepath).convert("RGB")
        width, height = image.size
        
        # 1. Detect Layout
        predictions = predictor([image])[0]
        
        # 2. Extract Bounding Boxes
        raw_bboxes = predictions.bboxes
        bboxes = [b.bbox for b in raw_bboxes]

        # 3. Identify Gutter logic
        left_boxes = [b for b in bboxes if (b[0] + b[2])/2 < width/2]
        right_boxes = [b for b in bboxes if (b[0] + b[2])/2 >= width/2]

        if not left_boxes or not right_boxes:
            print(f"   ⚠️ Layout unclear. Falling back to center split.")
            split_x = width // 2
        else:
            max_left_x = max([b[2] for b in left_boxes])
            min_right_x = min([b[0] for b in right_boxes])
            split_x = int((max_left_x + min_right_x) / 2)
            print(f"   ✂️  Cutting at x={split_x}")

        # 4. Crop & Save (WITH HIGH QUALITY FIX)
        left_img = image.crop((0, 0, split_x, height))
        right_img = image.crop((split_x, 0, width, height))

        base = Path(filename).stem
        
        # quality=100: Max quality
        # subsampling=0: No color compression (keeps text sharp)
        left_img.save(Path(OUTPUT_DIR) / f"{base}_L.jpg", quality=100, subsampling=0)
        right_img.save(Path(OUTPUT_DIR) / f"{base}_R.jpg", quality=100, subsampling=0)
        
        print(f"   ✅ Saved {base}_L.jpg and {base}_R.jpg (High Res)")

    except Exception as e:
        print(f"   ❌ Error processing {filename}: {e}")

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print("🚀 Loading Surya Detection Model...")
    predictor = DetectionPredictor()

    for filename in HEAVY_FILES:
        smart_split(filename, predictor)
    
    del predictor
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()