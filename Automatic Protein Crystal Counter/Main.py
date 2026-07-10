import os
import cv2
import numpy as np
import pandas as pd
from cellpose import models

# ==========================================
# 1. HARDCODED LAB CONFIGURATION LOOKUP TABLES
# ==========================================
# Estimated average crystal width in PIXELS per magnification setting
# Cellpose natively works best when objects are roughly 30 pixels wide
MAGNIFICATION_DIAMETERS = {
    "10x": 15,  # Crystals look smaller
    "20x": 30,  # Standard baseline
    "40x": 60  # Crystals look larger
}

# Hemocytometer specifications to calculate volumetric density/concentration
# depth_mm: Depth of the chamber
# frame_area_mm2: Total field of view area in square mm captured by your camera at that magnification.
# Note: You can fine-tune 'frame_area_mm2' by looking at your microscope's field-of-view spec sheet.
HEMOCYTOMETER_SPECS = {
    "Neubauer": {
        "depth_mm": 0.1,
        "frame_area_mm2": 1.0  # Assumes the camera image spans a 1x1mm grid area
    },
    "Fuchs": {
        "depth_mm": 0.2,
        "frame_area_mm2": 0.5  # Example frame area adjustment
    }
}

# ==========================================
# 2. INITIALIZE PIPELINE
# ==========================================
input_folder = "./crystal_images"  # Change to your folder path
output_csv = "./crystal_density_results.csv"

# Ensure input directory exists
if not os.path.exists(input_folder):
    os.makedirs(input_folder)
    print(f"Created '{input_folder}' folder. Please place your JPEGs/PNGs inside it and rerun.")
    exit()

print("Initializing Cellpose Deep Learning Generalist Model (cyto3)...")
# Initialize the generalized CNN. Will use GPU automatically if available.
model = models.CellposeModel(gpu=True, model_type='cyto3')

results_list = []

# ==========================================
# 3. BATCH PROCESSING LOOP
# ==========================================
print(f"Scanning '{input_folder}' for images...")
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files

    print(f"\nProcessing file: {filename}")

    # Parse filename tokens (Expected structure: Slide_Mag_Details.ext)
    try:
        parts = filename.split('_')
        slide_type = parts[0]  # e.g., "Neubauer"
        magnification = parts[1]  # e.g., "10x"
        sample_id = "_".join(parts[2:]).split('.')[0]  # Remainder of name
    except IndexError:
        print(f"⚠️ Skipping '{filename}'. Does not follow standard naming convention: 'Slide_Mag_Details.jpg'")
        continue

    # Validate parsed parameters against our hardcoded configurations
    if slide_type not in HEMOCYTOMETER_SPECS or magnification not in MAGNIFICATION_DIAMETERS:
        print(f"⚠️ Skipping '{filename}'. Unsupported slide type ({slide_type}) or magnification ({magnification}).")
        continue

    # Load configuration values
    expected_diameter = MAGNIFICATION_DIAMETERS[magnification]
    slide_depth = HEMOCYTOMETER_SPECS[slide_type]["depth_mm"]
    view_area = HEMOCYTOMETER_SPECS[slide_type]["frame_area_mm2"]

    # Read image using OpenCV
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)

    # Run CNN inference to generate masks
    # channels=[0,0] instructs Cellpose to handle standard grayscale or color images without fluorescent nuclear stains
    masks, flows, styles = model.eval(image, diameter=expected_diameter, channels=[0, 0])

    # Count the unique objects found by the AI (highest assigned mask ID)
    crystal_count = masks.max()

    # ==========================================
    # 4. MATH FORMULATION: CONCENTRATION
    # ==========================================
    # Total volume imaged = Area of view (mm^2) * Chamber Depth (mm)
    total_volume_mm3 = view_area * slide_depth

    # Convert cubic millimeters (mm^3) to milliliters (mL) -> 1 mm^3 = 0.001 mL
    total_volume_ml = total_volume_mm3 * 0.001

    # Concentration calculation (Crystals per mL)
    if total_volume_ml > 0:
        concentration_per_ml = crystal_count / total_volume_ml
    else:
        concentration_per_ml = 0

    print(f"   -> Found {crystal_count} crystals.")
    print(f"   -> Calculated Concentration: {concentration_per_ml:.2e} crystals/mL")

    # ==========================================
    # NEW STEP: GENERATE TRAINING LABELS AUTOMATICALLY
    # ==========================================
    labels_folder = "./automated_labels"
    os.makedirs(labels_folder, exist_ok=True)

    # Cellpose assigns sequential integers to masks (0=bg, 1=crystal1, 2=crystal2...).
    # We turn any pixel > 0 into a solid white pixel (255) to make a clean binary mask.
    binary_mask = np.where(masks > 0, 255, 0).astype(np.uint8)

    # Save the mask using a matching filename prefix so the annotation software links them
    mask_filename = f"mask_{os.path.splitext(filename)[0]}.png"
    cv2.imwrite(os.path.join(labels_folder, mask_filename), binary_mask)
    print(f"   -> Saved automated annotation to {labels_folder}/{mask_filename}")

    # Append data dictionary for final compilation
    results_list.append({
        "Filename": filename,
        "Sample_ID": sample_id,
        "Slide_Type": slide_type,
        "Magnification": magnification,
        "Crystal_Count": crystal_count,
        "Total_Volume_Imaged_mL": total_volume_ml,
        "Concentration_Crystals_per_mL": concentration_per_ml
    })

# ==========================================
# 5. DATA EXPORTATION
# ==========================================
if results_list:
    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False)
    print(f"\n Processing complete! All metrics successfully saved to: {output_csv}")
else:
    print("\n No valid images were processed.")