"""
V2 Crystal Counter — Cellpose-based protein crystal detection for hemocytometer slides.

Detects crystals in hemocytometer images, counts them, and calculates
volumetric concentration (crystals/mL). Also exports binary masks for
downstream annotation/training.

Usage:
    python count_crystals.py
    python count_crystals.py --input ./my_images --output results.csv
    python count_crystals.py --labels ./my_labels --no-labels

Expected filename convention:  Slide_Mag_Details.ext
    e.g.  Neubauer_20x_sampleA.jpg
"""

import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd
from cellpose import models

# Version-safe GPU detection import
try:
    from cellpose.core import use_gpu
except ImportError:
    # Fallback for API differences across versions
    def use_gpu():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False


# ==========================================
# 1. HARDCODED LAB CONFIGURATION LOOKUP TABLES
# ==========================================
# Estimated average crystal width in PIXELS per magnification setting.
# Cellpose natively works best when objects are roughly 30 pixels wide.
MAGNIFICATION_DIAMETERS = {
    "10x": 15,  # Crystals look smaller
    "20x": 30,  # Standard baseline
    "40x": 60,  # Crystals look larger
}

# Hemocytometer specifications to calculate volumetric density/concentration.
# depth_mm: Depth of the chamber
# frame_area_mm2: Total field-of-view area (mm^2) captured by your camera at that magnification.
# Note: You can fine-tune 'frame_area_mm2' using your microscope's field-of-view spec sheet.
HEMOCYTOMETER_SPECS = {
    "Neubauer": {
        "depth_mm": 0.1,
        "frame_area_mm2": 1.0,  # Assumes the camera image spans a 1x1mm grid area
    },
    "Fuchs": {
        "depth_mm": 0.2,
        "frame_area_mm2": 0.5,  # Example frame area adjustment
    },
}


# ==========================================
# 2. ARGUMENT PARSING
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and count protein crystals in hemocytometer images using Cellpose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", default="./crystal_images",
        help="Folder containing input images (.png/.jpg/.jpeg).",
    )
    parser.add_argument(
        "--output", "-o", default="./crystal_density_results.csv",
        help="Path for the output CSV file.",
    )
    parser.add_argument(
        "--labels", "-l", default="./automated_labels",
        help="Folder to save automated binary mask annotations.",
    )
    parser.add_argument(
        "--model", default="cyto3",
        help="Cellpose model type to use.",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Disable saving of automated binary mask annotations.",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU usage even if a GPU is available.",
    )
    return parser.parse_args()


# ==========================================
# 3. VERSION-SAFE CELLPOSE EVAL WRAPPER
# ==========================================
def run_eval(model, image, diameter):
    """
    Run model.eval and handle API differences across Cellpose versions.
    Some versions return (masks, flows, styles); others return
    (masks, flows, styles, diams). We only need the masks.
    """
    try:
        # Newer versions may not accept the 'channels' argument.
        result = model.eval(image, diameter=diameter, channels=[0, 0])
    except TypeError:
        result = model.eval(image, diameter=diameter)

    # First element is always the masks regardless of return length.
    masks = result[0]
    return masks


# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    args = parse_args()

    input_folder = args.input
    output_csv = args.output
    labels_folder = args.labels
    save_labels = not args.no_labels

    # Ensure input directory exists
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created '{input_folder}' folder. "
              f"Please place your JPEGs/PNGs inside it and rerun.")
        sys.exit(0)

    # GPU detection
    gpu_available = False if args.cpu else use_gpu()
    print(f"GPU available / enabled: {gpu_available}")

    print(f"Initializing Cellpose model ('{args.model}')...")
    print("NOTE: Please know that when you run this script for the first time the model weights will download (~hundreds of MB). "
          "This requires an internet connection and may take a while.")
    try:
        model = models.CellposeModel(gpu=gpu_available, model_type=args.model)
    except Exception as e:
        print(f"ERROR: Failed to initialize Cellpose model: {e}")
        sys.exit(1)

    if save_labels:
        os.makedirs(labels_folder, exist_ok=True)

    results_list = []

    # ==========================================
    # BATCH PROCESSING LOOP
    # ==========================================
    print(f"Scanning '{input_folder}' for images...")
    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue  # Skip non-image files

        print(f"\nProcessing file: {filename}")

        # Parse filename tokens (Expected structure: Slide_Mag_Details.ext)
        parts = filename.split("_")
        if len(parts) < 3:
            print(f"[SKIP] '{filename}' does not follow naming convention: "
                  f"'Slide_Mag_Details.ext'")
            continue

        slide_type = parts[0]                          # e.g., "Neubauer"
        magnification = parts[1]                        # e.g., "10x"
        sample_id = "_".join(parts[2:]).rsplit(".", 1)[0]  # Remainder of name

        # Validate parsed parameters against hardcoded configurations
        if slide_type not in HEMOCYTOMETER_SPECS or magnification not in MAGNIFICATION_DIAMETERS:
            print(f"[SKIP] '{filename}': unsupported slide type "
                  f"({slide_type}) or magnification ({magnification}).")
            continue

        # Load configuration values
        expected_diameter = MAGNIFICATION_DIAMETERS[magnification]
        slide_depth = HEMOCYTOMETER_SPECS[slide_type]["depth_mm"]
        view_area = HEMOCYTOMETER_SPECS[slide_type]["frame_area_mm2"]

        # Read image using OpenCV
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[SKIP] Could not read '{filename}' (corrupt or unsupported file).")
            continue

        # Run CNN inference (version-safe)
        try:
            masks = run_eval(model, image, expected_diameter)
        except Exception as e:
            print(f"[SKIP] Inference failed for '{filename}': {e}")
            continue

        # Count unique objects found (highest assigned mask ID). Cast to plain int.
        crystal_count = int(masks.max())

        # ==========================================
        # MATH FORMULATION: CONCENTRATION
        # ==========================================
        total_volume_mm3 = view_area * slide_depth      # mm^3
        total_volume_ml = total_volume_mm3 * 0.001       # 1 mm^3 = 0.001 mL

        concentration_per_ml = (crystal_count / total_volume_ml) if total_volume_ml > 0 else 0

        print(f"   -> Found {crystal_count} crystals.")
        print(f"   -> Calculated Concentration: {concentration_per_ml:.2e} crystals/mL")

        # ==========================================
        # GENERATE TRAINING LABELS (optional)
        # ==========================================
        if save_labels:
            # Turn any pixel > 0 into white (255) for a clean binary mask.
            binary_mask = np.where(masks > 0, 255, 0).astype(np.uint8)
            mask_filename = f"mask_{os.path.splitext(filename)[0]}.png"
            cv2.imwrite(os.path.join(labels_folder, mask_filename), binary_mask)
            print(f"   -> Saved automated annotation to {labels_folder}/{mask_filename}")

        # Append data for final compilation
        results_list.append({
            "Filename": filename,
            "Sample_ID": sample_id,
            "Slide_Type": slide_type,
            "Magnification": magnification,
            "Crystal_Count": crystal_count,
            "Total_Volume_Imaged_mL": total_volume_ml,
            "Concentration_Crystals_per_mL": concentration_per_ml,
        })

    # ==========================================
    # DATA EXPORTATION
    # ==========================================
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(output_csv, index=False)
        print(f"\nProcessing complete! Metrics saved to: {output_csv}")
    else:
        print("\nNo valid images were processed.")


if __name__ == "__main__":
    main()