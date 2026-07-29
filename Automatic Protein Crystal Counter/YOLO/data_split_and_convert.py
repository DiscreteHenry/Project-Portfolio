import os
import shutil
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
RAW_LABELME_DIR = 'data/raw_labelme_images'  # Your folder with image/json pairs
PROCESSED_DIR = 'data/processed'
TRAIN_IMAGES_DIR = os.path.join(PROCESSED_DIR, 'images', 'train')
VAL_IMAGES_DIR = os.path.join(PROCESSED_DIR, 'images', 'val')
TEST_IMAGES_DIR = os.path.join(PROCESSED_DIR, 'images', 'test')  # Optional
TRAIN_ANNOTATIONS_JSON = os.path.join(PROCESSED_DIR, 'annotations', 'instances_train.json')
VAL_ANNOTATIONS_JSON = os.path.join(PROCESSED_DIR, 'annotations', 'instances_val.json')
TEST_ANNOTATIONS_JSON = os.path.join(PROCESSED_DIR, 'annotations', 'instances_test.json')  # Optional

# --- Update converter.py path ---
# Adjust this path if your converter.py is located elsewhere
CONVERTER_SCRIPT = 'scripts/model_dev/utilities/converter.py'

# ROOT_PREFIX: This is CRUCIAL. It's the *absolute path* to where the images will be
# located on the machine where you run the training.
# E.g., if you run training from 'your_project_root/' and images are in 'data/processed/images/train/',
# then ROOT_PREFIX for training data would be 'your_project_root/data/processed/images/train/'.
# Make sure this matches the eventual location for the training environment.
# For simplicity, we'll set it relative to the project root assuming training is run from there.
# You might need to adjust this on your training server.
# For now, let's assume the converter will prepend the current working directory.
# We'll set this *inside* the converter script for each run.
# Alternatively, modify converter.py to take ROOT_PREFIX as an argument.
# For this example, we'll modify the converter's script content dynamically.

TEST_SIZE = 0.15  # 15% for validation
VAL_SIZE = 0.15  # 15% for validation (remaining from non-test)
# If TEST_SIZE is 0.15, then remaining is 0.85. If VAL_SIZE is 0.15 of that remaining:
# VAL_SIZE_FROM_TOTAL = 0.85 * 0.15 approx 0.1275.
# Let's do a simple split: train, val, test = 70%, 15%, 15%

# --- Create directories ---
os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)  # Optional
os.makedirs(os.path.join(PROCESSED_DIR, 'annotations'), exist_ok=True)

# --- Collect image and annotation paths ---
image_files = []
for fname in os.listdir(RAW_LABELME_DIR):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
        json_path = os.path.join(RAW_LABELME_DIR, fname.rsplit('.', 1)[0] + '.json')
        if os.path.exists(json_path):
            image_files.append(fname)
        else:
            print(f"Warning: No JSON found for {fname}, skipping.")

if not image_files:
    print("No image-JSON pairs found in raw_labelme_images. Please check the directory and file names.")
    exit()

# --- Split data ---
print(f"Found {len(image_files)} image-JSON pairs.")
train_val_files, test_files = train_test_split(image_files, test_size=TEST_SIZE, random_state=42)
train_files, val_files = train_test_split(train_val_files, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                          random_state=42)  # Adjust val_size relative to the remaining

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


# --- Function to process a split ---
def process_split(files, image_dest_dir, output_json_path, root_prefix_for_converter):
    print(f"\nProcessing {os.path.basename(image_dest_dir)} split...")
    # Copy images and JSONs
    labelme_temp_dir = 'temp_labelme_jsons_for_converter'
    os.makedirs(labelme_temp_dir, exist_ok=True)

    for fname in tqdm(files, desc=f"Copying files to {os.path.basename(image_dest_dir)}"):
        src_image_path = os.path.join(RAW_LABELME_DIR, fname)
        src_json_path = os.path.join(RAW_LABELME_DIR, fname.rsplit('.', 1)[0] + '.json')

        # Copy image to its final destination
        shutil.copy(src_image_path, image_dest_dir)
        # Copy JSON to temp dir for converter
        shutil.copy(src_json_path, labelme_temp_dir)

    # --- Dynamically modify converter.py for this run ---
    # This is a hack because your converter.py expects hardcoded paths.
    # A better solution would be to modify converter.py to accept arguments.
    converter_content = open(CONVERTER_SCRIPT).read()

    modified_converter_content = converter_content
    modified_converter_content = modified_converter_content.replace(
        f"LABELME_DIR = \"{RAW_LABELME_DIR}\"",  # Assuming this is the default in converter.py
        f"LABELME_DIR = \"{os.path.abspath(labelme_temp_dir)}\""
    )
    modified_converter_content = modified_converter_content.replace(
        f"OUT_JSON = \"output.json\"",  # Assuming this is the default
        f"OUT_JSON = \"{os.path.abspath(output_json_path)}\""
    )
    modified_converter_content = modified_converter_content.replace(
        f"ROOT_PREFIX = \"/path/to/coyote/image/folder\"",  # Assuming this is the default
        f"ROOT_PREFIX = \"{os.path.abspath(image_dest_dir)}/\""  # Add trailing slash for consistency
    )

    # Save modified converter to a temporary file
    temp_converter_script = "temp_converter.py"
    with open(temp_converter_script, "w") as f:
        f.write(modified_converter_content)

    print(f"Running converter for {os.path.basename(image_dest_dir)}...")
    os.system(f"python {temp_converter_script}")
    print(f"Conversion complete for {os.path.basename(image_dest_dir)}.")

    # Clean up
    shutil.rmtree(labelme_temp_dir)
    os.remove(temp_converter_script)


# --- Process each split ---
current_project_root = os.getcwd()  # Get the current working directory of your script

# Ensure ROOT_PREFIX for converter is correct. This assumes your training script
# will be run from `your_project_root/`
process_split(train_files, TRAIN_IMAGES_DIR, TRAIN_ANNOTATIONS_JSON,
              os.path.join(current_project_root, TRAIN_IMAGES_DIR, ''))
process_split(val_files, VAL_IMAGES_DIR, VAL_ANNOTATIONS_JSON, os.path.join(current_project_root, VAL_IMAGES_DIR, ''))
if TEST_SIZE > 0:
    process_split(test_files, TEST_IMAGES_DIR, TEST_ANNOTATIONS_JSON,
                  os.path.join(current_project_root, TEST_IMAGES_DIR, ''))

print("\nData splitting and conversion complete!")