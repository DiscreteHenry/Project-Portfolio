"""Prepare a tiled YOLO dataset from LabelMe rectangles and run tiled inference."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Annotation:
    json_path: Path
    image_path: Path
    boxes: list[Box]
    ignored_labels: dict[str, int]


def image_for_labelme(json_path: Path, data: dict) -> Path | None:
    """Resolve the image explicitly named in LabelMe, then safely try its stem."""
    named_path = data.get("imagePath")
    if named_path:
        candidate = json_path.parent / named_path
        if candidate.is_file():
            return candidate
    matches = [path for path in json_path.parent.glob(f"{json_path.stem}.*")
               if path.suffix.lower() in IMAGE_EXTENSIONS]
    return matches[0] if len(matches) == 1 else None


def rectangle_from_shape(shape: dict, image_width: int, image_height: int) -> Box | None:
    """Normalize a LabelMe rectangle, clipping it to the source image."""
    if shape.get("shape_type", "rectangle") != "rectangle":
        return None
    points = shape.get("points", [])
    if len(points) != 2 or any(len(point) < 2 for point in points):
        return None
    x_values = sorted((float(points[0][0]), float(points[1][0])))
    y_values = sorted((float(points[0][1]), float(points[1][1])))
    box = Box(max(0.0, x_values[0]), max(0.0, y_values[0]),
              min(float(image_width), x_values[1]), min(float(image_height), y_values[1]))
    return box if box.width > 0 and box.height > 0 else None


def yolo_row(box: Box, image_width: int, image_height: int, class_id: int = 0) -> str:
    """Convert pixel xyxy coordinates to one normalized YOLO detection row."""
    x_center, y_center = box.center
    return (f"{class_id} {x_center / image_width:.8f} {y_center / image_height:.8f} "
            f"{box.width / image_width:.8f} {box.height / image_height:.8f}")


def boxes_for_tile(boxes: Iterable[Box], x_offset: int, y_offset: int,
                   tile_width: int, tile_height: int) -> list[Box]:
    """Keep boxes centered in a tile, then convert and clip them to tile coordinates."""
    selected: list[Box] = []
    x_end, y_end = x_offset + tile_width, y_offset + tile_height
    for box in boxes:
        center_x, center_y = box.center
        if not (x_offset <= center_x < x_end and y_offset <= center_y < y_end):
            continue
        clipped = Box(max(box.x1, x_offset) - x_offset, max(box.y1, y_offset) - y_offset,
                      min(box.x2, x_end) - x_offset, min(box.y2, y_end) - y_offset)
        if clipped.width > 0 and clipped.height > 0:
            selected.append(clipped)
    return selected


def tile_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or not 0 <= overlap < tile_size:
        raise ValueError("tile_size must be positive and overlap must be in [0, tile_size).")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, length - tile_size + 1, stride))
    final_position = length - tile_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def load_annotations(raw_dir: Path, class_label: str, strict: bool) -> tuple[list[Annotation], list[dict]]:
    import cv2

    annotations: list[Annotation] = []
    report: list[dict] = []
    for json_path in sorted(raw_dir.glob("*.json")):
        entry = {"json": str(json_path), "valid": False, "errors": [], "warnings": [], "boxes": 0}
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            image_path = image_for_labelme(json_path, data)
            if image_path is None:
                raise ValueError("could not resolve exactly one matching image")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"OpenCV could not read {image_path.name}")
            image_height, image_width = image.shape[:2]
            if data.get("imageWidth") and data["imageWidth"] != image_width:
                entry["warnings"].append("imageWidth differs from image pixels")
            if data.get("imageHeight") and data["imageHeight"] != image_height:
                entry["warnings"].append("imageHeight differs from image pixels")
            boxes, ignored = [], {}
            for index, shape in enumerate(data.get("shapes", [])):
                label = shape.get("label", "")
                if label != class_label:
                    ignored[label] = ignored.get(label, 0) + 1
                    continue
                box = rectangle_from_shape(shape, image_width, image_height)
                if box is None:
                    entry["warnings"].append(f"shape {index}: invalid rectangle")
                else:
                    boxes.append(box)
            entry.update(valid=True, image=str(image_path), boxes=len(boxes), ignored_labels=ignored)
            annotations.append(Annotation(json_path, image_path, boxes, ignored))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            entry["errors"].append(str(error))
        report.append(entry)
    failures = [entry for entry in report if not entry["valid"]]
    if strict and failures:
        raise ValueError(f"{len(failures)} invalid LabelMe file(s); see validation report")
    return annotations, report


def write_overlays(annotations: list[Annotation], output_dir: Path, count: int, seed: int) -> None:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    selection = random.Random(seed).sample(annotations, min(count, len(annotations)))
    for annotation in selection:
        image = cv2.imread(str(annotation.image_path))
        for box in annotation.boxes:
            cv2.rectangle(image, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / f"original_{annotation.image_path.name}"), image)


def safe_id(path: Path) -> str:
    return "".join(character if character.isalnum() else "_" for character in path.stem)


def prepare_dataset(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np

    raw_dir, output_dir = Path(args.raw_dir), Path(args.output_dir)
    annotations, report = load_annotations(raw_dir, args.class_label, args.strict)
    if not annotations:
        raise ValueError("No valid annotated images were found.")
    has_existing_dataset = any((output_dir / name).exists() for name in ("images", "labels"))
    if has_existing_dataset and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains dataset folders; pass --overwrite to recreate them.")
    if has_existing_dataset:
        for child in (output_dir / "images", output_dir / "labels"):
            if child.exists():
                shutil.rmtree(child)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    ordered = annotations[:]
    random.Random(args.seed).shuffle(ordered)
    total = len(ordered)
    if total < 3:
        raise ValueError("At least three labelled source images are required for train/val/test splits.")
    train_count = max(1, round(total * args.train_fraction))
    val_count = max(1, round(total * args.val_fraction))
    if train_count + val_count >= total:
        val_count = max(1, total - train_count - 1)
    if train_count + val_count >= total:
        train_count = total - val_count - 1
    splits = {"train": ordered[:train_count], "val": ordered[train_count:train_count + val_count],
              "test": ordered[train_count + val_count:]}
    if not all(splits.values()):
        raise ValueError("Each split must receive at least one source image; adjust fractions or add annotations.")

    manifest, metadata = [], []
    for split, source_annotations in splits.items():
        image_dir, label_dir = output_dir / "images" / split, output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for source_index, annotation in enumerate(source_annotations):
            source = cv2.imread(str(annotation.image_path))
            height, width = source.shape[:2]
            source_key = f"{safe_id(annotation.image_path)}_{source_index:03d}"
            manifest.append({"source_id": source_key, "split": split, "image": str(annotation.image_path),
                             "json": str(annotation.json_path), "boxes": len(annotation.boxes)})
            for y_offset in tile_positions(height, args.tile_size, args.overlap):
                for x_offset in tile_positions(width, args.tile_size, args.overlap):
                    tile = np.zeros((args.tile_size, args.tile_size, 3), dtype=source.dtype)
                    crop = source[y_offset:min(height, y_offset + args.tile_size), x_offset:min(width, x_offset + args.tile_size)]
                    tile[:crop.shape[0], :crop.shape[1]] = crop
                    tile_name = f"{source_key}__x{x_offset}_y{y_offset}.png"
                    tile_boxes = boxes_for_tile(annotation.boxes, x_offset, y_offset, args.tile_size, args.tile_size)
                    cv2.imwrite(str(image_dir / tile_name), tile)
                    (label_dir / f"{Path(tile_name).stem}.txt").write_text(
                        "\n".join(yolo_row(box, args.tile_size, args.tile_size) for box in tile_boxes), encoding="utf-8")
                    metadata.append({"tile": str(image_dir / tile_name), "label": str(label_dir / f"{Path(tile_name).stem}.txt"),
                                     "source_id": source_key, "source_image": str(annotation.image_path), "split": split,
                                     "x_offset": x_offset, "y_offset": y_offset, "tile_size": args.tile_size,
                                     "source_width": width, "source_height": height, "box_count": len(tile_boxes)})
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with (output_dir / "tile_metadata.jsonl").open("w", encoding="utf-8") as handle:
        handle.writelines(json.dumps(entry) + "\n" for entry in metadata)
    write_overlays(annotations, output_dir / "verification" / "originals", args.overlay_count, args.seed)
    write_tile_overlays(metadata, output_dir / "verification" / "tiles", args.overlay_count, args.seed)
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: crystal\n", encoding="utf-8")
    print(f"Prepared {len(metadata)} tiles from {total} images at {output_dir}")


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        _, center_x, center_y, box_width, box_height = map(float, line.split()[:5])
        center_x, box_width = center_x * width, box_width * width
        center_y, box_height = center_y * height, box_height * height
        boxes.append(Box(center_x - box_width / 2, center_y - box_height / 2,
                         center_x + box_width / 2, center_y + box_height / 2))
    return boxes


def write_tile_overlays(metadata: list[dict], output_dir: Path, count: int, seed: int) -> None:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    selection = random.Random(seed + 1).sample(metadata, min(count, len(metadata)))
    for entry in selection:
        image = cv2.imread(entry["tile"])
        for box in read_yolo_boxes(Path(entry["label"]), image.shape[1], image.shape[0]):
            cv2.rectangle(image, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 255, 0), 2)
        cv2.imwrite(str(output_dir / Path(entry["tile"]).name), image)


def nms(detections: list[tuple[Box, float]], iou_threshold: float) -> list[tuple[Box, float]]:
    kept: list[tuple[Box, float]] = []
    for box, confidence in sorted(detections, key=lambda item: item[1], reverse=True):
        if all(intersection_over_union(box, existing) < iou_threshold for existing, _ in kept):
            kept.append((box, confidence))
    return kept


def intersection_over_union(first: Box, second: Box) -> float:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    union = first.width * first.height + second.width * second.height - intersection_width * intersection_height
    return (intersection_width * intersection_height / union) if union else 0.0


def labelme_shape(box: Box, label: str = "Protein Crystal", confidence: float | None = None) -> dict:
    """Create an editable LabelMe rectangle shape, optionally retaining confidence metadata."""
    flags = {} if confidence is None else {"pseudo_confidence": round(confidence, 6)}
    return {
        "label": label,
        "points": [[round(box.x1, 3), round(box.y1, 3)], [round(box.x2, 3), round(box.y2, 3)]],
        "group_id": None,
        "description": "Pseudo-label: review and correct this rectangle.",
        "shape_type": "rectangle",
        "flags": flags,
    }


def tiled_predictions(model, image, tile_size: int, overlap: int,
                      confidence: float, nms_iou: float) -> list[tuple[Box, float]]:
    """Predict on overlapping tiles and return globally deduplicated detections."""
    import numpy as np

    detections: list[tuple[Box, float]] = []
    height, width = image.shape[:2]
    for y_offset in tile_positions(height, tile_size, overlap):
        for x_offset in tile_positions(width, tile_size, overlap):
            tile = np.zeros((tile_size, tile_size, 3), dtype=image.dtype)
            crop = image[y_offset:min(height, y_offset + tile_size), x_offset:min(width, x_offset + tile_size)]
            tile[:crop.shape[0], :crop.shape[1]] = crop
            result = model.predict(tile, conf=confidence, verbose=False)[0]
            for xyxy, score, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(class_id) != 0:
                    continue
                detections.append((Box(float(xyxy[0] + x_offset), float(xyxy[1] + y_offset),
                                       float(xyxy[2] + x_offset), float(xyxy[3] + y_offset)), float(score)))
    return nms(detections, nms_iou)


def pseudo_label(args: argparse.Namespace) -> None:
    """Generate editable full-image LabelMe JSON pseudo-labels from an Ultralytics model."""
    import cv2
    import numpy as np
    from ultralytics import YOLO

    source_dir, output_dir = Path(args.source_dir), Path(args.output_dir)
    image_paths = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No supported images found in {source_dir}")
    model = YOLO(args.weights)
    image_dir, labelme_dir = output_dir / "images", output_dir / "labelme"
    image_dir.mkdir(parents=True, exist_ok=True)
    labelme_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    csv_path = output_dir / "detections.csv"
    with metadata_path.open("w", encoding="utf-8") as metadata_handle, csv_path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=["image", "id", "x1", "y1", "x2", "y2", "confidence"])
        writer.writeheader()
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue
            height, width = image.shape[:2]
            detections = tiled_predictions(model, image, args.tile_size, args.overlap, args.confidence, args.nms_iou)
            shapes = []
            for index, (box, score) in enumerate(detections, start=1):
                clipped = Box(max(0.0, box.x1), max(0.0, box.y1), min(float(width), box.x2), min(float(height), box.y2))
                if clipped.width <= 0 or clipped.height <= 0:
                    continue
                shapes.append(labelme_shape(clipped, args.class_label, score))
                writer.writerow({"image": image_path.name, "id": index, **asdict(clipped), "confidence": f"{score:.6f}"})
                cv2.rectangle(image, (round(clipped.x1), round(clipped.y1)), (round(clipped.x2), round(clipped.y2)), (0, 255, 0), 2)
                cv2.putText(image, f"{index}:{score:.2f}", (round(clipped.x1), max(15, round(clipped.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            shutil.copy2(image_path, image_dir / image_path.name)
            shutil.copy2(image_path, labelme_dir / image_path.name)
            labelme_data = {"version": "5.3.1", "flags": {}, "shapes": shapes,
                            "imagePath": image_path.name, "imageData": None,
                            "imageHeight": height, "imageWidth": width}
            json_path = labelme_dir / f"{image_path.stem}.json"
            json_path.write_text(json.dumps(labelme_data, indent=2), encoding="utf-8")
            cv2.imwrite(str(visualization_dir / image_path.name), image)
            metadata_handle.write(json.dumps({"image": str(image_path), "labelme_json": str(json_path),
                                               "visualization": str(visualization_dir / image_path.name),
                                               "detections": len(shapes), "confidence": args.confidence,
                                               "tile_size": args.tile_size, "overlap": args.overlap,
                                               "nms_iou": args.nms_iou}) + "\n")
    print(f"Wrote editable LabelMe pseudo-labels for {len(image_paths)} images to {labelme_dir}")


def infer(args: argparse.Namespace) -> None:
    import cv2
    import numpy as np
    from ultralytics import YOLO

    image_path, output_dir = Path(args.image), Path(args.output_dir)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read {image_path}")
    height, width = image.shape[:2]
    model = YOLO(args.weights)
    detections: list[tuple[Box, float]] = []
    for y_offset in tile_positions(height, args.tile_size, args.overlap):
        for x_offset in tile_positions(width, args.tile_size, args.overlap):
            tile = np.zeros((args.tile_size, args.tile_size, 3), dtype=image.dtype)
            crop = image[y_offset:min(height, y_offset + args.tile_size), x_offset:min(width, x_offset + args.tile_size)]
            tile[:crop.shape[0], :crop.shape[1]] = crop
            result = model.predict(tile, conf=args.confidence, verbose=False)[0]
            for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                if int(class_id) != 0:
                    continue
                box = Box(float(xyxy[0] + x_offset), float(xyxy[1] + y_offset),
                          float(xyxy[2] + x_offset), float(xyxy[3] + y_offset))
                detections.append((box, float(confidence)))
    unique = nms(detections, args.nms_iou)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = image.copy()
    with (output_dir / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "x1", "y1", "x2", "y2", "confidence"])
        writer.writeheader()
        for index, (box, confidence) in enumerate(unique, start=1):
            writer.writerow({"id": index, **asdict(box), "confidence": f"{confidence:.6f}"})
            cv2.rectangle(annotated, (round(box.x1), round(box.y1)), (round(box.x2), round(box.y2)), (0, 255, 0), 2)
            cv2.putText(annotated, str(index), (round(box.x1), max(15, round(box.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(annotated, f"Unique crystals: {len(unique)}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "annotated_result.png"), annotated)
    (output_dir / "inference_metadata.json").write_text(json.dumps({"image": str(image_path), "raw_detections": len(detections), "unique_crystals": len(unique), "tile_size": args.tile_size, "overlap": args.overlap, "nms_iou": args.nms_iou}, indent=2), encoding="utf-8")
    print(f"Found {len(unique)} unique crystals. Results: {output_dir}")


def train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=str(Path(args.dataset) / "dataset.yaml"), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=args.project, name=args.name, seed=args.seed)


def add_common_tiling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate LabelMe and create split tiled YOLO data.")
    prepare.add_argument("--raw-dir", default=DEFAULT_ROOT / "data" / "raw_labelme_images", type=Path)
    prepare.add_argument("--output-dir", default=DEFAULT_ROOT / "data" / "processed_seed_data", type=Path)
    prepare.add_argument("--class-label", default="Protein Crystal")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--train-fraction", type=float, default=0.7)
    prepare.add_argument("--val-fraction", type=float, default=0.15)
    prepare.add_argument("--overlay-count", type=int, default=5)
    prepare.add_argument("--strict", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")
    add_common_tiling_arguments(prepare)
    prepare.set_defaults(handler=prepare_dataset)
    training = commands.add_parser("train", help="Train Ultralytics YOLO.")
    training.add_argument("--dataset", default=DEFAULT_ROOT / "data" / "processed_seed_data", type=Path)
    training.add_argument("--model", default="yolo11n.pt")
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--imgsz", type=int, default=512)
    training.add_argument("--batch", type=int, default=8)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--project", default="runs/crystal_detection")
    training.add_argument("--name", default="baseline")
    training.set_defaults(handler=train)
    inference = commands.add_parser("infer", help="Run tiled full-image inference and global NMS.")
    inference.add_argument("--weights", required=True)
    inference.add_argument("--image", required=True, type=Path)
    inference.add_argument("--output-dir", required=True, type=Path)
    inference.add_argument("--confidence", type=float, default=0.25)
    inference.add_argument("--nms-iou", type=float, default=0.5)
    add_common_tiling_arguments(inference)
    inference.set_defaults(handler=infer)
    pseudo = commands.add_parser("pseudo-label", help="Create editable LabelMe JSON predictions for unlabeled images.")
    pseudo.add_argument("--weights", required=True)
    pseudo.add_argument("--source-dir", default=DEFAULT_ROOT / "data" / "unlabeled_pool", type=Path)
    pseudo.add_argument("--output-dir", default=DEFAULT_ROOT / "data" / "pseudo_labels" / "iter1", type=Path)
    pseudo.add_argument("--class-label", default="Protein Crystal")
    pseudo.add_argument("--confidence", type=float, default=0.15)
    pseudo.add_argument("--nms-iou", type=float, default=0.5)
    add_common_tiling_arguments(pseudo)
    pseudo.set_defaults(handler=pseudo_label)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "train_fraction", 0) + getattr(args, "val_fraction", 0) >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1.")
    args.handler(args)


if __name__ == "__main__":
    main()
