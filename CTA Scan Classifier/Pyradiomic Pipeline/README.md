# CTA Scan Image Classifier Using Pyradiomics

 The code base implements a machine learning pipeline to automatically assess the visual quality of CTA (CT Angiography) scans using handcrafted radiomic features extracted from multiple 2D slices per scan. The classifier predicts whether a scan is usable (1) or unusable (0) based on attributes like sharpness, contrast, motion artifacts, and field of view (FOV).

Current Model Performance
Random Forest:
 Validation Accuracy 0.84
LogisticRegression:
 Validation Accuracy 0.80

---

## Project Overview

* **Input**: CTA scans in `.nii`, `.nii.gz`, or DICOM folder format
* **Preprocessing**:    Extracts 12 representative 2D slices per scan (10 axial, 1 coronal, 1 sagittal)
                        Applies 2D radiomic feature extraction per slice (e.g., first-order, GLCM, GLDM)
* **Model**: Random Forest Classifier trained on radiomic feature vectors
* **Output**: Binary prediction of scan quality: 1 is usable, 0 is unusable

---

## Project Structure

```
├── Training Scripts/
│   └── (model)_training_script.py              # Full feature extraction and training pipeline
│
├── Inference Scripts/
│   └── (model)_inference_script.py             # Inference script for new scans
│
├── Classifier Models/
│   └── trained_(model)_model.pkl               # Trained model for inference
│
├── Extracted Features CSVs/
│   ├── full_(model)_features.csv               # Raw extracted radiomic features
│   ├── reorganized_full_(model)_features.csv   # Long-format per-slice features
│   └── (model)_feature_importance.csv          # Feature importance by slice
│
├── Radiomic Labels.csv                         # Ground truth labels
└── README.md                                   # Project documentation
```

---

## Requirements

SimpleITK, pyradiomics, numpy, pandas, scikit-learn
joblib, tqdm, matplotlib, pathlib, concurrent.futures
Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Instructions for Training the Model

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
python "training_script.py"
```

This will:

* Load scans and labels
* Train a CNN on 12 slices per scan
* Save the best model as `trained_radiomics_model.pkl`
* Plot and save training/validation loss curves

---

## Instructions for Running Inference

To predict the quality of a new scan:

```bash
python "inference_script.py"
```

Or use it in Python:

```python
from predict_cta_score import predict_quality
score = predict_quality("trained_radiomics_model.pkl", "path/to/scan.nii.gz")
print(f"Predicted score: {score:.2f}")
```

---

## Metrics

* Accuracy, F1 Score, Precision, Recall
* Balanced accuracy for imbalanced datasets
* Confusion matrix breakdown
* Feature importances ranked by slice

---

## Next Steps

* Make a userfriendly GUI for easy Scan/Directory selection
* Add in data analysis for troubleshooting and feature analysis
* Automate said data analysis
* Use LightGBM or XGBoost for better performance
* Try feature selection (e.g., mutual info or LASSO)
* Explore combining radiomic + CNN deep features
* Add Grad-CAM or SHAP-style interpretation for radiomics

---
