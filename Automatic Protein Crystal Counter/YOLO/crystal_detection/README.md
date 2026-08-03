# Crystal detection with LabelMe and YOLO

This reproducible pipeline prepares one-class crystal-detection data from LabelMe rectangles, trains Ultralytics YOLO, and runs tiled inference on full microscope images.

## Inspected repository layout

The annotated files are in `../data/raw_labelme_images`. The current LabelMe files name their paired PNGs in `imagePath` and contain rectangles labelled `Protein Crystal` and `Center Grid`. The default configuration converts only `Protein Crystal` to YOLO class `0` (`crystal`) and reports the ignored grid rectangles. Unlabeled images are in `../data/unlabeled_pool`; this pipeline never treats them as negative training examples.

## Setup

```powershell
cd "Automatic Protein Crystal Counter/YOLO/crystal_detection"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Validate, split, tile, and visualize

Source images are split before tiling, so tiles from one original can never leak into a different split. This requires at least three labelled originals. The deterministic allocator preserves one source image for every split when the dataset is small.

```powershell
python crystal_yolo.py prepare --strict --overwrite --seed 42
```

The command writes the requested YOLO structure under `../data/processed_seed_data`:

```text
images/train  images/val  images/test
labels/train  labels/val  labels/test
```

It also writes `validation_report.json`, `split_manifest.json`, `tile_metadata.jsonl`, `dataset.yaml`, and label overlays in `verification/originals` and `verification/tiles`. Tile offsets and original-image paths are preserved in `tile_metadata.jsonl`. Tiles are 512 px with 128 px overlap by default; boxes are assigned by center and clipped at tile edges.

To use another LabelMe class, set `--class-label` explicitly. All source, output, split, seed, and tiling paths/settings are command-line arguments.

## Train

```powershell
python crystal_yolo.py train --dataset ../data/processed_seed_data --model yolo11n.pt --epochs 100 --imgsz 512 --batch 8
```

Ultralytics places training artifacts in `runs/crystal_detection/baseline` by default. Adjust the model, batch size, and epochs for your GPU memory and dataset size.

## Full-image tiled inference

```powershell
python crystal_yolo.py infer --weights runs/crystal_detection/baseline/weights/best.pt --image ../data/unlabeled_pool/example.png --output-dir inference/example
```

Inference tiles the image, maps tile predictions back to global coordinates, globally suppresses duplicates with NMS, and writes `annotated_result.png`, `detections.csv`, and `inference_metadata.json`. The final CSV has one row per unique detected crystal.

## Generate editable pseudo-labels

After the first weak model is trained, generate predictions for the unlabeled pool:

```powershell
python crystal_yolo.py pseudo-label --weights runs/crystal_detection/baseline/weights/best.pt --source-dir ../data/unlabeled_pool --output-dir ../data/pseudo_labels/iter1
```

This uses the same 512 px tiles, 128 px overlap, global coordinate conversion, and global NMS as inference. It writes copied images and editable LabelMe JSON files under `data/pseudo_labels/iter1/labelme`, plus annotated previews, a CSV, and metadata. Open the copied image/JSON pairs from that `labelme` folder in LabelMe and correct the boxes. Then copy the corrected image/JSON pairs into `data/raw_labelme_images` (or another input directory) and rerun `prepare` to train the next iteration. The default confidence is intentionally 0.15 so low-confidence candidates are available for human review; increase it if the overlays are too noisy.

## Tests

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
```

The tests cover LabelMe/YOLO box conversion and overlap-tile coordinate behavior.
