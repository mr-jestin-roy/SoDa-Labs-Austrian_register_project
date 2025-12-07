# HTR Stream Preprocessing for Sterb Buch Images
# Processes scraped images from matricula-online with CLAHE enhancement
# Replaces original images in-place for optimal HTR processing

# 1. Import libraries
import cv2
import numpy as np
import os
from glob import glob

# 2. Set input folder path for scraped images
input_folder = (
    "Well_behaved_sterb_buch_images/Althofen_A050271"  # Scraped images folder
)

# 3. Get all image paths in the folder
image_paths = glob(os.path.join(input_folder, "*.jpg"))

# 4. Setup CLAHE with optimized parameters for HTR preprocessing
clahe = cv2.createCLAHE(
    clipLimit=3.0, tileGridSize=(8, 8)
)  # Optimized for text enhancement

print(f"📊 Found {len(image_paths)} images to process...")
print("🔄 Starting CLAHE enhancement and in-place replacement...")

# 5. Process each image in the folder and replace original
for idx, img_path in enumerate(image_paths, start=1):
    # Extract the base file name (without extension)
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"[{idx}/{len(image_paths)}] Processing: {base_filename}")

    # Load grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Skipping invalid image: {img_path}")
        continue

    # Apply CLAHE enhancement
    enhanced = clahe.apply(img)

    # Replace the original image with enhanced version (in-place)
    cv2.imwrite(img_path, enhanced)
    print(f"✅ Enhanced and replaced: {base_filename}")

print(f"\n🎉 Successfully processed {len(image_paths)} images!")
print("📁 All images have been enhanced with CLAHE and replaced in-place.")

# 6. Verification - Check a few processed images
print("\n🔍 Verification: Checking processed images...")
sample_images = image_paths[:3]  # Check first 3 images

for img_path in sample_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        print(f"✅ {os.path.basename(img_path)}: {img.shape} - Successfully processed")
    else:
        print(f"❌ {os.path.basename(img_path)}: Failed to load")

print("\n📋 Summary:")
print(f"   • Input folder: {input_folder}")
print(f"   • Images processed: {len(image_paths)}")
print(f"   • Enhancement: CLAHE (clipLimit=3.0, tileGridSize=8x8)")
print(f"   • Output: In-place replacement (original files overwritten)")
