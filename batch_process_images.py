#!/usr/bin/env python3
"""
Batch process multiple images with Unified Detector and extract text regions
"""

import os
import subprocess
import json
import cv2
import numpy as np
from pathlib import Path
import glob
import sys

def get_python_executable():
    """Get the current Python executable path"""
    return sys.executable

def run_unified_detector(image_path, output_dir):
    """
    Run Unified Detector inference on a single image
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save outputs
    
    Returns:
        Path to generated JSONL file
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output paths
    jsonl_path = os.path.join(output_dir, "detection.jsonl")
    vis_dir = os.path.join(output_dir, "visualizations")
    
    # Get the Python executable from current environment
    python_exec = get_python_executable()
    
    # Run Unified Detector with current environment
    cmd = [
        python_exec, "-m", "official.projects.unified_detector.run_inference",
        f"--gin_file=official/projects/unified_detector/configs/gin_files/unified_detector_model.gin",
        f"--ckpt_path=ckpt",
        f"--img_file={image_path}",
        f"--output_path={jsonl_path}",
        f"--vis_dir={vis_dir}"
    ]
    
    print(f"Running Unified Detector on {os.path.basename(image_path)}...")
    print(f"Using Python: {python_exec}")
    
    try:
        # Run with current environment variables
        env = os.environ.copy()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(f"✅ Detection completed for {os.path.basename(image_path)}")
        return jsonl_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running detection on {image_path}")
        print(f"Error: {e.stderr}")
        return None

def extract_text_regions_from_file(jsonl_path, image_path, output_base_dir, levels=['word', 'line', 'paragraph']):
    """
    Extract text regions from a specific JSONL file
    
    Args:
        jsonl_path: Path to JSONL detection file
        image_path: Path to original image
        output_base_dir: Base directory for extracted regions
        levels: List of extraction levels
    """
    
    if not os.path.exists(jsonl_path):
        print(f"❌ JSONL file not found: {jsonl_path}")
        return
    
    # Load the JSONL file
    with open(jsonl_path, 'r') as f:
        data = json.loads(f.read())
    
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"📷 Processing extractions for {os.path.basename(image_path)}")
    
    # Extract for each level
    for level in levels:
        level_output_dir = os.path.join(output_base_dir, level)
        os.makedirs(level_output_dir, exist_ok=True)
        
        count = extract_single_level(data, image, level_output_dir, level)
        print(f"  📦 Extracted {count} {level} regions")

def extract_single_level(data, image, output_dir, level):
    """Extract regions for a single level (word/line/paragraph)"""
    
    annotations = data.get('annotations', [])
    count = 0
    
    for annotation in annotations:
        paragraphs = annotation.get('paragraphs', [])
        
        for paragraph in paragraphs:
            if level == 'paragraph':
                # For paragraph level, collect all vertices from all lines
                all_vertices = []
                lines = paragraph.get('lines', [])
                for line in lines:
                    words = line.get('words', [])
                    for word in words:
                        vertices = word.get('vertices', [])
                        all_vertices.extend(vertices)
                
                if all_vertices:
                    count = save_region(image, all_vertices, output_dir, level, count)
            
            else:
                # Process lines
                lines = paragraph.get('lines', [])
                for line in lines:
                    if level == 'line':
                        # For line level, collect all vertices from all words in the line
                        all_vertices = []
                        words = line.get('words', [])
                        for word in words:
                            vertices = word.get('vertices', [])
                            all_vertices.extend(vertices)
                        
                        if all_vertices:
                            count = save_region(image, all_vertices, output_dir, level, count)
                    
                    elif level == 'word':
                        # For word level, process each word individually
                        words = line.get('words', [])
                        for word in words:
                            vertices = word.get('vertices', [])
                            if vertices:
                                count = save_region(image, vertices, output_dir, level, count)
    
    return count

def save_region(image, vertices, output_dir, level, count):
    """Save a cropped region from vertices"""
    if len(vertices) < 3:  # Need at least 3 points
        return count
    
    # Convert vertices to numpy array
    points = np.array(vertices)
    
    # Get bounding box
    x_min, y_min = np.min(points, axis=0).astype(int)
    x_max, y_max = np.max(points, axis=0).astype(int)
    
    # Add padding
    padding = 5
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_max = min(image.shape[0], y_max + padding)
    
    # Skip if bounding box is too small
    if (x_max - x_min) < 10 or (y_max - y_min) < 10:
        return count
    
    # Crop the region
    cropped = image[y_min:y_max, x_min:x_max]
    
    if cropped.size > 0:
        # Save the cropped image
        output_path = os.path.join(output_dir, f"{level}_{count:04d}.jpg")
        cv2.imwrite(output_path, cropped)
        count += 1
    
    return count

def batch_process_images(input_dir, output_base_dir):
    """
    Process all images in input directory
    
    Args:
        input_dir: Directory containing input images
        output_base_dir: Base directory for all outputs
    """
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.tif', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"🚀 Found {len(image_files)} images to process")
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        image_name = Path(image_path).stem
        print(f"\n📸 Processing image {i}/{len(image_files)}: {image_name}")
        
        # Create output directory for this image
        image_output_dir = os.path.join(output_base_dir, image_name)
        
        # Run Unified Detector
        jsonl_path = run_unified_detector(image_path, image_output_dir)
        
        if jsonl_path and os.path.exists(jsonl_path):
            # Extract text regions
            regions_output_dir = os.path.join(image_output_dir, "extracted_regions")
            extract_text_regions_from_file(jsonl_path, image_path, regions_output_dir)
        else:
            print(f"❌ Skipping extraction for {image_name} due to detection failure")
    
    print(f"\n🎉 Batch processing complete!")
    print(f"📁 All outputs saved in: {output_base_dir}")

def main():
    """Main function"""
    
    # Configuration
    input_dir = "test_images/input_images"
    output_base_dir = "test_images/batch_outputs"
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    print("🔄 Starting batch processing...")
    print(f"📂 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_base_dir}")
    
    # Run batch processing
    batch_process_images(input_dir, output_base_dir)

if __name__ == "__main__":
    main() 