import os
import shutil
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import the new COCO to YOLO converter
from scripts.model_dev.utilities.coco_to_yolo_txt import convert_coco_to_yolo

# --- Configuration ---
# 1. RAW_LABELME_DIR: This should point to your initial small set of manually labeled images.
#    For the *first run* in the pseudo-labeling workflow, this will be 'data/raw_labelme_images'.
#    In *subsequent iterations*, this will be 'data/combined_labeled_images_iterN'.
RAW_LABELME_DIR = 'data/raw_labelme_images'

# 2. PROCESSED_DIR: This is where the COCO-like datasets (images and annotations) will be
#    created for training.
#    For the *first run*, this will be 'data/processed_seed_data'.
#    In *subsequent iterations*, this will be 'data/processed_iterN_data'.
PROCESSED_DIR = 'data/processed_seed_data'

# --- Internal Paths (Derived from PROCESSED_DIR) ---
TRAIN_IMAGES_DIR = os.path.join(PROCESSED_DIR, 'images', 'train')
VAL_IMAGES_DIR = os.path.join(PROCESSED_DIR, 'images', 'val')
TEST_IMAGES_DIR = os.path.join(PROCESSED_DIR, 'images', 'test')

# Also define paths for the 'labels' directories where YOLO .txt files will be saved
TRAIN_LABELS_DIR = os.path.join(PROCESSED_DIR, 'labels', 'train')
VAL_LABELS_DIR = os.path.join(PROCESSED_DIR, 'labels', 'val')
TEST_LABELS_DIR = os.path.join(PROCESSED_DIR, 'labels', 'test')

# Paths for the intermediate COCO JSONs
TRAIN_ANNOTATIONS_JSON = os.path.join(PROCESSED_DIR, 'annotations', 'instances_train.json')
VAL_ANNOTATIONS_JSON = os.path.join(PROCESSED_DIR, 'annotations', 'instances_val.json')
TEST_ANNOTATIONS_JSON = os.path.join(PROCESSED_DIR, 'annotations', 'instances_test.json')

# --- Converter Script Path (for LabelMe to COCO) ---
# Adjust this path if your converter.py is located elsewhere relative to the project root.
CONVERTER_SCRIPT = 'scripts/model_dev/utilities/converter.py'

# --- Dataset Splitting Ratios ---
SMALL_DATASET_THRESHOLD = 5  # If total images <= 5, all go to train.
TEST_SIZE_RATIO = 0.15  # 15% for test split (if total images > threshold)
VAL_SIZE_RATIO = 0.15  # 15% for validation split (if total images > threshold)


# --- Create directories ---
def setup_directories():
    print(f"Setting up directories in '{PROCESSED_DIR}'...")
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, 'annotations'), exist_ok=True)

    # Create placeholder 'labels' directories for YOLO .txt files
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
    os.makedirs(VAL_LABELS_DIR, exist_ok=True)
    os.makedirs(TEST_LABELS_DIR, exist_ok=True)
    print("Directories created/ensured.")


# --- Function to process a split (copy files, run LabelMe->COCO, then COCO->YOLO TXT) ---
def process_split(files, image_dest_dir, output_coco_json_path, output_yolo_labels_dir, root_prefix_for_converter,
                  split_name):
    print(f"\nProcessing {split_name} split...")

    # Clean up previous YOLO labels for this split, if any
    if os.path.exists(output_yolo_labels_dir):
        for f in os.listdir(output_yolo_labels_dir):
            if f.endswith('.txt'):
                os.remove(os.path.join(output_yolo_labels_dir, f))

    labelme_temp_dir = f'temp_labelme_jsons_for_converter_{split_name}'
    os.makedirs(labelme_temp_dir, exist_ok=True)

    for fname in tqdm(files, desc=f"Copying files to {split_name}"):
        src_image_path = os.path.join(RAW_LABELME_DIR, fname)
        src_json_path = os.path.join(RAW_LABELME_DIR, fname.rsplit('.', 1)[0] + '.json')

        shutil.copy(src_image_path, image_dest_dir)
        shutil.copy(src_json_path, labelme_temp_dir)

    # --- Step 1: Run LabelMe JSON to COCO JSON conversion ---
    # Read the content of converter.py to dynamically modify it
    try:
        with open(CONVERTER_SCRIPT, 'r', encoding='utf-8') as f:
            converter_content = f.read()
    except FileNotFoundError:
        print(f"Error: converter.py not found at '{CONVERTER_SCRIPT}'. Please check the path.")
        return

    modified_converter_content = converter_content
    modified_converter_content = modified_converter_content.replace(
        "LABELME_DIR = \"/some/path/to/your/labelme/annotations/\"",  # Assumed placeholder in converter.py
        f"LABELME_DIR = \"{os.path.abspath(labelme_temp_dir).replace(os.sep, '/')}\""
        # Use absolute path with forward slashes
    )
    modified_converter_content = modified_converter_content.replace(
        "OUT_JSON = \"/some/path/to/your/output_coco_annotations.json\"",  # Assumed placeholder
        f"OUT_JSON = \"{os.path.abspath(output_coco_json_path).replace(os.sep, '/')}\""
    )
    modified_converter_content = modified_converter_content.replace(
        "ROOT_PREFIX = \"/some/path/to/your/image/folder/on/training/machine/\"",  # Assumed placeholder
        f"ROOT_PREFIX = \"{os.path.abspath(image_dest_dir).replace(os.sep, '/')}/\""  # Add trailing slash
    )

    temp_converter_script = "temp_converter.py"
    with open(temp_converter_script, "w", encoding='utf-8') as f:
        f.write(modified_converter_content)

    print(f"Running LabelMe to COCO converter for {split_name}...")
    try:
        os.system(f"python {temp_converter_script}")
        print(f"LabelMe to COCO conversion complete for {split_name}.")
    except Exception as e:
        print(f"Error running LabelMe to COCO converter for {split_name}: {e}")
    finally:
        shutil.rmtree(labelme_temp_dir)
        os.remove(temp_converter_script)

    # --- Step 2: Run COCO JSON to YOLO TXT conversion ---
    print(f"Running COCO to YOLO .txt converter for {split_name}...")
    convert_coco_to_yolo(output_coco_json_path, output_yolo_labels_dir, image_dest_dir)
    print(f"COCO to YOLO .txt conversion complete for {split_name}.")


# --- Main Script Execution ---
if __name__ == '__main__':
    setup_directories()

    image_files = []
    for fname in os.listdir(RAW_LABELME_DIR):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            json_path = os.path.join(RAW_LABELME_DIR, fname.rsplit('.', 1)[0] + '.json')
            if os.path.exists(json_path):
                image_files.append(fname)
            else:
                print(f"Warning: No JSON found for {fname} in '{RAW_LABELME_DIR}', skipping.")

    if not image_files:
        print(f"No image-JSON pairs found in '{RAW_LABELME_DIR}'. Please check the directory and file names.")
        # Ensure COCO JSONs and YOLO TXT directories are created even if no files
        current_project_root = os.getcwd()
        process_split([], TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_JSON, TRAIN_LABELS_DIR,
                      os.path.join(current_project_root, TRAIN_IMAGES_DIR), "train")
        process_split([], VAL_IMAGES_DIR, VAL_ANNOTATIONS_JSON, VAL_LABELS_DIR,
                      os.path.join(current_project_root, VAL_IMAGES_DIR), "val")
        process_split([], TEST_IMAGES_DIR, TEST_ANNOTATIONS_JSON, TEST_LABELS_DIR,
                      os.path.join(current_project_root, TEST_IMAGES_DIR), "test")
        exit()

    print(f"Found {len(image_files)} image-JSON pairs in '{RAW_LABELME_DIR}'.")

    train_files = []
    val_files = []
    test_files = []

    if len(image_files) <= SMALL_DATASET_THRESHOLD:
        print(f"Dataset has {len(image_files)} images, which is <= {SMALL_DATASET_THRESHOLD}.")
        print("Putting all images into the training set for initial weak model.")
        train_files = image_files
    else:
        print(f"Dataset has {len(image_files)} images, which is > {SMALL_DATASET_THRESHOLD}.")
        print(
            f"Splitting into train/val/test with ratios: {1 - TEST_SIZE_RATIO - VAL_SIZE_RATIO:.0%} / {VAL_SIZE_RATIO:.0%} / {TEST_SIZE_RATIO:.0%}.")

        train_val_files, test_files = train_test_split(image_files, test_size=TEST_SIZE_RATIO, random_state=42)
        val_relative_size = VAL_SIZE_RATIO / (1.0 - TEST_SIZE_RATIO)
        train_files, val_files = train_test_split(train_val_files, test_size=val_relative_size, random_state=42)

    print(f"Split results - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    current_project_root = os.getcwd()

    # Process each split, now generating both COCO JSON and YOLO TXT files
    process_split(train_files, TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_JSON, TRAIN_LABELS_DIR,
                  os.path.join(current_project_root, TRAIN_IMAGES_DIR), "train")
    process_split(val_files, VAL_IMAGES_DIR, VAL_ANNOTATIONS_JSON, VAL_LABELS_DIR,
                  os.path.join(current_project_root, VAL_IMAGES_DIR), "val")
    process_split(test_files, TEST_IMAGES_DIR, TEST_ANNOTATIONS_JSON, TEST_LABELS_DIR,
                  os.path.join(current_project_root, TEST_IMAGES_DIR), "test")

    print("\nData splitting and all conversions (LabelMe -> COCO -> YOLO .txt) complete!")