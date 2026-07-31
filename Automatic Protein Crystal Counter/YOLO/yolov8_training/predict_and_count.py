from ultralytics import YOLO
import os
# import cv2 # Not strictly needed for pseudo-label generation
import json
from tqdm import tqdm

# --- Configuration for generating pseudo-labels ---
# Set these for each iteration!
MODEL_PATH = 'crystal_counting_yolov8/pseudo_label_iter0/weights/best.pt' # Path to your *weak* trained model
IMAGE_SOURCE_PATH = 'data/unlabeled_pool/batch_1' # Directory of unlabeled images for *this iteration*
PSEUDO_LABEL_OUTPUT_DIR = 'pseudo_label_runs/iter1' # Base directory for this iteration's outputs
PSEUDO_LABEL_TXT_DIR = os.path.join(PSEUDO_LABEL_OUTPUT_DIR, 'labels') # Where YOLO TXT files will be saved
PSEUDO_LABEL_VIS_DIR = os.path.join(PSEUDO_LABEL_OUTPUT_DIR, 'visualizations') # Where annotated images will be saved

CONFIDENCE_THRESHOLD = 0.25 # Lower this to get more predictions, even if uncertain. Human will filter.
IOU_THRESHOLD = 0.7         # NMS IoU threshold. Keep it reasonable.

def generate_pseudo_labels():
    print(f"Loading model from {MODEL_PATH} for pseudo-label generation...")
    model = YOLO(MODEL_PATH)

    os.makedirs(PSEUDO_LABEL_TXT_DIR, exist_ok=True)
    os.makedirs(PSEUDO_LABEL_VIS_DIR, exist_ok=True) # For visual checking

    # Get list of image files to process
    image_files = []
    if os.path.isdir(IMAGE_SOURCE_PATH):
        for fname in os.listdir(IMAGE_SOURCE_PATH):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_files.append(os.path.join(IMAGE_SOURCE_PATH, fname))
    else:
        print(f"Error: IMAGE_SOURCE_PATH '{IMAGE_SOURCE_PATH}' is not a valid directory.")
        return

    print(f"Generating pseudo-labels on {len(image_files)} image(s) from {IMAGE_SOURCE_PATH}...")

    # Perform prediction
    # save=True will save annotated images to PSEUDO_LABEL_VIS_DIR (via project/name)
    # save_txt=True will save YOLO TXT files to PSEUDO_LABEL_TXT_DIR (via project/name/labels)
    results = model.predict(
        source=IMAGE_SOURCE_PATH,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        save=True,
        save_txt=True,
        save_conf=True, # Saves confidence in TXT, useful for review
        project=PSEUDO_LABEL_VIS_DIR, # Base for annotated images
        name='pseudo_labels_vis', # Subfolder within project
        labels=True, # Ensures bounding boxes are labeled on image
        hide_labels=False, hide_conf=False,
        verbose=False
    )
    # Note: YOLO will create subdirectories like PSEUDO_LABEL_VIS_DIR/pseudo_labels_vis/labels
    # and PSEUDO_LABEL_VIS_DIR/pseudo_labels_vis/images

    print(f"\nPseudo-label generation complete.")
    print(f"YOLO TXT predictions saved to: {PSEUDO_LABEL_VIS_DIR}/pseudo_labels_vis/labels")
    print(f"Visualizations saved to: {PSEUDO_LABEL_VIS_DIR}/pseudo_labels_vis/")

if __name__ == '__main__':
    generate_pseudo_labels()