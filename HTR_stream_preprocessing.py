# 1. Import libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

# 2. Set input and output folder paths
input_folder = "happy path_Sterb Buch/inputs/Yearwise cutoff comparison"  # Replace with your actual input folder
output_folder = (
    "happy path_Sterb Buch/outputs"  # Folder where enhanced images will be saved
)

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# 3. Get all image paths in the folder (PG)
image_paths = glob(os.path.join(input_folder, "*.jpg"))

# 4. Setup CLAHE with more aggressive parameters for visible enhancement
clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))  # More aggressive settings

# 5. Process each image in the folder
for idx, img_path in enumerate(image_paths, start=1):
    # Extract the base file name (without extension)
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"[{idx}] Processing input image: {base_filename}")

    # Load grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping invalid image: {img_path}")
        continue

    # Apply CLAHE
    enhanced = clahe.apply(img)

    # Save the enhanced image with a modified file name
    output_filename = os.path.join(output_folder, f"{base_filename}_clahe_enhanced.png")
    cv2.imwrite(output_filename, enhanced)
    print(f"[{idx}] Saved: {output_filename}")

# 6. Show input vs enhanced comparison for all images after processing
print("\nDisplaying comparison of input vs CLAHE-enhanced images...")

for idx, img_path in enumerate(image_paths, start=1):
    # Extract the base file name (without extension)
    base_filename = os.path.splitext(os.path.basename(img_path))[0]

    # Load original image in COLOR
    original_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    original_color_rgb = cv2.cvtColor(
        original_color, cv2.COLOR_BGR2RGB
    )  # Convert for matplotlib

    # Load CLAHE enhanced grayscale image
    enhanced_path = os.path.join(output_folder, f"{base_filename}_clahe_enhanced.png")
    enhanced_gray = cv2.imread(enhanced_path, cv2.IMREAD_GRAYSCALE)

    if enhanced_gray is None:
        print(f"Warning: Could not load enhanced image: {enhanced_path}")
        continue

    # Create simple side-by-side comparison
    plt.figure(figsize=(16, 8))

    # Left: Original in COLOR
    plt.subplot(1, 2, 1)
    plt.title(f"Original (Color)\n{os.path.basename(img_path)}", fontsize=12)
    plt.imshow(original_color_rgb)
    plt.axis("off")

    # Right: CLAHE Enhanced in GRAYSCALE
    plt.subplot(1, 2, 2)
    plt.title(
        f"CLAHE Optimized (Grayscale)\n{base_filename}_clahe_enhanced.png", fontsize=12
    )
    plt.imshow(enhanced_gray, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
