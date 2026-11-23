#!/usr/bin/env python3
"""
Fast Batch Chandra OCR (Direct API Version)
Bypasses the CLI to prevent timeouts and allow custom prompts.
"""

import requests
import base64
import os
import time
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
INPUT_DIR = "./high_res_lr_splits"  # Your input folder
OUTPUT_DIR = "./chandra_output"                       # Your output folder
VLLM_URL = "http://localhost:8010/v1/chat/completions"
WORKERS = 1  # Set to 2 since we are using prefix caching and direct API, set to 1 for handling heavy pages

# --- YOUR CUSTOM PROMPT ---
# This matches the instruction you wanted to add
CUSTOM_PROMPT = (
    "Transcribe this table from the Austrian Parish Records (Pfarramt Pernegg a.d Mur). "
    "Use layout detection to identify the grid. "
    "Output ONLY valid HTML code with <table> tags. Do not wrap in markdown."
)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def encode_image(image_path):
    """Encodes an image file to a base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def process_image(image_path):
    filename = image_path.name
    
    # 1. Prepare Output Path
    # Handle filenames with spaces safely
    safe_stem = image_path.stem.replace(" ", "_")
    output_subdir = Path(OUTPUT_DIR) / safe_stem
    output_subdir.mkdir(parents=True, exist_ok=True)
    html_path = output_subdir / f"{safe_stem}.html"
    
    # Skip if already done
    if html_path.exists():
        logger.info(f"⏩ Skipping {filename} (Already exists)")
        return

    logger.info(f"🚀 Sending {filename}...")

    try:
        # 2. Direct API Call (Bypasses CLI overhead)
        base64_img = encode_image(image_path)
        
        payload = {
            "model": "chandra",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        # The text instruction comes FIRST for better Prefix Caching
                        {"type": "text", "text": CUSTOM_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }
            ],
            "max_tokens": 8192,  # Large window for full tables
            "temperature": 0.1   # Low temp for accuracy
        }
        
        start_ts = time.time()
        
        # 3. Send Request
        response = requests.post(VLLM_URL, json=payload, timeout=600)
        response.raise_for_status()
        
        # 4. Save Result
        content = response.json()['choices'][0]['message']['content']
        
        # Strip markdown code blocks if the model adds them (e.g. ```html ... ```)
        if content.startswith("```html"): 
            content = content[7:]
        if content.endswith("```"): 
            content = content[:-3]
        
        with open(html_path, "w", encoding='utf-8') as f:
            f.write(content)
            
        duration = time.time() - start_ts
        logger.info(f"✅ Finished {filename} in {duration:.1f}s")

    except Exception as e:
        logger.error(f"❌ Failed {filename}: {e}")

def main():
    # Find all images
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        logger.error(f"Input directory not found: {INPUT_DIR}")
        return

    images = sorted(list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.jpeg")) + list(input_path.rglob("*.png")))
    logger.info(f"Found {len(images)} images. Starting processing with {WORKERS} workers.")
    
    # Run in parallel
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        executor.map(process_image, images)

if __name__ == "__main__":
    main()