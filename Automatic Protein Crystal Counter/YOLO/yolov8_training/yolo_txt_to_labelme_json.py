import os
import cv2
import json
from tqdm import tqdm


def convert_yolo_to_labelme(image_dir, yolo_labels_dir, output_labelme_dir, class_names):
    """
    Converts YOLO format (.txt) bounding box annotations to LabelMe JSON format.

    Args:
        image_dir (str): Directory containing the images.
        yolo_labels_dir (str): Directory containing YOLO format .txt label files.
        output_labelme_dir (str): Directory to save LabelMe JSON files.
        class_names (list): List of class names corresponding to YOLO class IDs (e.g., ['crystal']).
    """
    os.makedirs(output_labelme_dir, exist_ok=True)

    yolo_files = [f for f in os.listdir(yolo_labels_dir) if f.endswith('.txt')]

    print(f"Converting {len(yolo_files)} YOLO .txt files to LabelMe .json...")

    for yolo_file in tqdm(yolo_files):
        base_name = os.path.splitext(yolo_file)[0]

        # Try to find the corresponding image (assuming common image extensions)
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            potential_image_path = os.path.join(image_dir, base_name + ext)
            if os.path.exists(potential_image_path):
                image_path = potential_image_path
                break

        if not image_path:
            print(f"Warning: Image not found for {yolo_file}. Skipping.")
            continue

        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            continue

        img_height, img_width, _ = img.shape

        labelme_data = {
            "version": "5.3.1",  # Use your LabelMe version
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,  # LabelMe often embeds image data, but not strictly needed for just correction
            "imageHeight": img_height,
            "imageWidth": img_width
        }

        with open(os.path.join(yolo_labels_dir, yolo_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id x_center y_center width height (and conf if saved)
                    class_id = int(parts[0])
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])

                    # Convert normalized YOLO format to pixel (x_min, y_min, x_max, y_max)
                    x_center_px = x_center_norm * img_width
                    y_center_px = y_center_norm * img_height
                    width_px = width_norm * img_width
                    height_px = height_norm * img_height

                    x1 = int(x_center_px - width_px / 2)
                    y1 = int(y_center_px - height_px / 2)
                    x2 = int(x_center_px + width_px / 2)
                    y2 = int(y_center_px + height_px / 2)

                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width - 1, x2)
                    y2 = min(img_height - 1, y2)

                    shape = {
                        "label": class_names[class_id],
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                    labelme_data["shapes"].append(shape)

        output_json_path = os.path.join(output_labelme_dir, base_name + '.json')
        with open(output_json_path, 'w') as f:
            json.dump(labelme_data, f, indent=4)

    print("Conversion complete.")


if __name__ == '__main__':
    # Example usage:
    # This script would typically be called within your iterative workflow.
    # The image_dir would be the current batch of unlabeled images.
    # The yolo_labels_dir would be where predict_pseudo_labels.py saved its TXT files.
    # The output_labelme_dir would be where you want to save JSONs for human correction.

    # Example placeholders:
    # IMAGE_BATCH_DIR = 'data/unlabeled_pool/batch_1/'
    # YOLO_TXT_OUTPUT_DIR = 'pseudo_label_runs/iter1/labels/' # Where YOLO saved its TXT predictions
    # LABELME_JSON_CORRECTION_DIR = 'data/pseudo_labeled_for_correction/iter1/'
    # CLASS_NAMES = ['crystal'] # Must match your class name in dataset.yaml

    # You'll need to replace these with actual paths for each iteration
    print("This script is meant to be called within the pseudo-labeling loop.")
    print("Please set IMAGE_BATCH_DIR, YOLO_TXT_OUTPUT_DIR, LABELME_JSON_CORRECTION_DIR, and CLASS_NAMES.")