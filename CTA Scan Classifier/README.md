# CTA Scan Quality Classifier

This project implements a deep learning pipeline to automatically assess the quality of CTA (CT Angiography) scans. The model predicts a score of either a 0 or 1 based on visual quality metrics such as sharpness, motion artifacts, contrast, and field of view (FOV).

---

## Project Overview

* **Input**: CTA scans in `.nii`, `.nii.gz`, or DICOM folder format
* **Preprocessing**: Extracts 12 representative 2D slices per scan (10 axial, 1 coronal, 1 sagittal)
* **Model**: 2D CNN feature extractor applied to each slice, with global feature aggregation and regression output
* **Output**: A single continuous quality score between 1 and 5

---

## Project Structure

```
├── train_cta_classifier.py       # Full training and evaluation pipeline
├── CTA Prediction Tool.ipynb          # Inference script for predicting scan quality
├── Labels.csv                    # Training labels (filename + quality score)
├── train.csv / val.csv           # Auto-generated splits
├── best_model.pt                 # Saved model weights
├── loss_curve.png                # Training/validation loss graph
└── README.md                     # Project description and usage guide
```

---

## Requirements

* Python 3.8+
* PyTorch
* nibabel
* pydicom
* opencv-python
* pandas
* tqdm
* pathlib
* scikit-learn
* matplotlib

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Ensure `Labels.csv` is formatted like:

```
filename,label
PREFFIR-11047 (...).nii.gz,4
PREFFIR-12002 (...).nii.gz,2
```

Each filename must map to a scan located at:

```
<root_dir>/<scan_id>/20/<filename>
```

Run the training script:

```bash
python train_cta_classifier.py
```

This will:

* Load scans and labels
* Train a CNN on 12 slices per scan
* Save the best model as `best_model.pt`
* Plot and save training/validation loss curves

---

## Predicting Scan Quality

To predict the quality of a new scan:

```bash
python predict_cta_score.py
```

Or use it in Python:

```python
from predict_cta_score import predict_quality
score = predict_quality("best_model.pt", "path/to/scan.nii.gz")
print(f"Predicted score: {score:.2f}")
```

---

## Metrics

* **MSELoss** is used during training
* **MAE** and **R²** are tracked per epoch
* Negative R² values indicate performance worse than a mean predictor

---

## Next Steps

* Improve model with pretrained ResNet encoders
* Try alternative slice selection or attention-based aggregation
* Refine labels for better consistency
* Add Grad-CAM visualization for interpretability

---
