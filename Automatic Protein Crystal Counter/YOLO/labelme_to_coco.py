import os
import json
from datetime import datetime
from tqdm import tqdm

LABELME_DIR = "data/raw_labelme_images"
OUT_JSON = "data/processed_seed_data/annotations/instances_train.json"

# what COCO will store in images[i]["file_name"]
# we will copy images into this folder so the paths are real
IMAGES_OUT_DIR = "data/processed_seed_data/images/train"


def convert_labelme_to_coco():
    coco = {
        "info": {
            "description": "Converted from LabelMe",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(IMAGES_OUT_DIR, exist_ok=True)

    label_to_cat_id = {}
    img_id = 0
    ann_id = 0
    cat_id = 0

    json_files = sorted([f for f in os.listdir(LABELME_DIR) if f.endswith(".json")])
    if not json_files:
        raise FileNotFoundError(f"No LabelMe .json files found in {LABELME_DIR}")

    for jf in tqdm(json_files, desc="LabelMe JSONs"):
        jp = os.path.join(LABELME_DIR, jf)
        with open(jp, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_path = data.get("imagePath")  # often just filename
        if not image_path:
            print(f"Skipping {jf}: missing imagePath")
            continue

        image_basename = os.path.basename(image_path)
        img_abs = os.path.join(LABELME_DIR, image_basename)
        if not os.path.exists(img_abs):
            print(f"Skipping {jf}: image file not found next to json: {img_abs}")
            continue

        w = data.get("imageWidth")
        h = data.get("imageHeight")
        if w is None or h is None:
            print(f"Skipping {jf}: missing imageWidth/imageHeight")
            continue

        img_id += 1

        # COCO typically uses relative paths
        coco_file_name = os.path.join(IMAGES_OUT_DIR, image_basename)

        coco["images"].append({
            "id": img_id,
            "width": w,
            "height": h,
            "file_name": coco_file_name,
        })

        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "rectangle":
                continue

            label = shape.get("label", "object")
            if label not in label_to_cat_id:
                cat_id += 1
                label_to_cat_id[label] = cat_id
                coco["categories"].append({
                    "id": cat_id,
                    "name": label,
                    "supercategory": "object",
                })

            pts = shape.get("points", [])
            if len(pts) != 2:
                continue

            (x1, y1), (x2, y2) = pts
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            bw = x_max - x_min
            bh = y_max - y_min
            if bw <= 0 or bh <= 0:
                continue

            ann_id += 1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": label_to_cat_id[label],
                "bbox": [x_min, y_min, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "segmentation": [],
            })

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print("Wrote:", OUT_JSON)
    print("images:", len(coco["images"]),
          "annotations:", len(coco["annotations"]),
          "categories:", len(coco["categories"]))


if __name__ == "__main__":
    convert_labelme_to_coco()