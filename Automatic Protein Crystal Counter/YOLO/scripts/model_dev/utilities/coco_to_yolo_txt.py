import json
import os
from tqdm import tqdm


def convert_coco_to_yolo(coco_json_path, output_yolo_dir, images_dir):
    """
    Converts COCO JSON annotations to YOLO format (.txt files).

    Args:
        coco_json_path (str): Path to the input COCO JSON file.
        output_yolo_dir (str): Directory where YOLO .txt files will be saved.
        images_dir (str): Directory containing the images referenced in COCO JSON.
                          This is needed to derive image filenames for YOLO .txt files
                          and to get image dimensions.
    """
    os.makedirs(output_yolo_dir, exist_ok=True)

    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: COCO JSON file not found at {coco_json_path}. Creating empty label directory.")
        return  # No annotations to convert

    if not coco_data.get('images') or not coco_data.get('annotations') or not coco_data.get('categories'):
        print(f"Warning: COCO JSON at {coco_json_path} appears empty or malformed. Creating empty label directory.")
        return

    # Create mappings for image IDs to filenames and category IDs to new class IDs (0, 1, 2...)
    # Use os.path.basename to get just the filename from potentially full paths in COCO JSON
    image_id_to_filename = {img['id']: os.path.basename(img['file_name']) for img in coco_data['images']}
    image_id_to_dims = {img['id']: {'width': img['width'], 'height': img['height']} for img in coco_data['images']}

    # YOLO requires class IDs to be 0-indexed and contiguous.
    # Map original COCO category IDs to new 0-indexed class IDs.
    category_id_map = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}

    # Store annotations per image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    print(f"Converting COCO JSON from {coco_json_path} to YOLO .txt format...")

    for img_id in tqdm(image_id_to_filename.keys(), desc="Processing images for YOLO conversion"):
        img_filename = image_id_to_filename[img_id]
        img_dims = image_id_to_dims.get(img_id)

        if not img_dims:
            print(f"Warning: Dimensions not found for image ID {img_id} ({img_filename}). Skipping its annotations.")
            continue

        img_width = img_dims['width']
        img_height = img_dims['height']

        yolo_lines = []
        for ann in annotations_by_image.get(img_id, []):
            cat_id = ann['category_id']
            if cat_id not in category_id_map:
                print(f"Warning: Category ID {cat_id} not found in categories. Skipping annotation.")
                continue

            yolo_class_id = category_id_map[cat_id]
            bbox = ann['bbox']  # [x_min, y_min, width, height]

            # Convert COCO bbox to YOLO format: [class_id x_center y_center width height] (normalized)
            x_min, y_min, w, h = bbox

            x_center = (x_min + w / 2) / img_width
            y_center = (y_min + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # Ensure values are clamped between 0 and 1
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm = max(0.0, min(1.0, w_norm))
            h_norm = max(0.0, min(1.0, h_norm))

            yolo_lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save to a .txt file named after the image
        yolo_filename = os.path.splitext(img_filename)[0] + '.txt'
        yolo_filepath = os.path.join(output_yolo_dir, yolo_filename)

        with open(yolo_filepath, 'w') as f:
            f.write('\n'.join(yolo_lines))

    print("COCO to YOLO .txt conversion complete.")


if __name__ == '__main__':
    # This script is meant to be imported and called by data_split_and_convert.py.
    # If testing independently, uncomment and set paths:
    # ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Assumes scripts/model_dev/utilities
    # coco_json_path = os.path.join(ROOT_DIR, 'data/processed_seed_data/annotations/instances_train.json')
    # output_yolo_dir = os.path.join(ROOT_DIR, 'data/processed_seed_data/labels/train')
    # images_dir = os.path.join(ROOT_DIR, 'data/processed_seed_data/images/train')
    # convert_coco_to_yolo(coco_json_path, output_yolo_dir, images_dir)
    print("This script is meant to be imported and called by data_split_and_convert.py")