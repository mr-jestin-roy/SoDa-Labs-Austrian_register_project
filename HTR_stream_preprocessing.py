# 1. Import libraries
import cv2
import matplotlib.pyplot as plt
import os
from glob import glob

# 2. Set input and output folder paths
input_folder = 'input_images'     # Replace with your actual input folder
output_folder = 'output_clahe'    # Folder where enhanced images will be saved

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# 3. Get all PNG image paths in the folder
image_paths = sorted(glob(os.path.join(input_folder, '*.png')))

# 4. Setup CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 5. Process each image in the folder
for idx, img_path in enumerate(image_paths, start=1):
    # Load grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping invalid image: {img_path}")
        continue

    # Apply CLAHE
    enhanced = clahe.apply(img)

    # Save the enhanced image
    output_filename = os.path.join(output_folder, f'sample{idx}.png')
    cv2.imwrite(output_filename, enhanced)
    print(f"[{idx}] Saved: {output_filename}")

# 6. Show input vs enhanced comparison for all images after processing
print("\nDisplaying comparison of input vs CLAHE-enhanced images...")

for idx, img_path in enumerate(image_paths, start=1):
    # Load original and processed image
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    enhanced = cv2.imread(os.path.join(output_folder, f'sample{idx}.png'), cv2.IMREAD_GRAYSCALE)

    # Create side-by-side plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Original {os.path.basename(img_path)}")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Enhanced sample{idx}.png")
    plt.imshow(enhanced, cmap='gray')
    plt.axis('off')

    plt.show()
