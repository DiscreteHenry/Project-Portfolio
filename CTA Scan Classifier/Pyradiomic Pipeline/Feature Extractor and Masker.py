from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Configure PyRadiomics
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()
extractor.settings['label'] = 1
extractor.settings['correctMask'] = False  # allow masks without strict segmentation
extractor.settings['force2D'] = True  # Required for 2D shape feature extraction

# Extract 12 representative slices from a 3D volume
def extract_12_slices(image):
    arr = sitk.GetArrayFromImage(image)  # shape: [Z, Y, X]
    z_slices = np.linspace(0, arr.shape[0] - 1, 10, dtype=int)
    axial = [arr[z, :, :] for z in z_slices]
    coronal = arr[arr.shape[0] // 2, :, :]
    sagittal = arr[:, :, arr.shape[2] // 2]
    return axial + [coronal, sagittal]

# Convert 2D slice to image and mask
def convert_to_image_and_mask(slice2d):
    slice3d = slice2d[np.newaxis, :, :]  # shape becomes [1, Y, X]

    image = sitk.GetImageFromArray(slice3d.astype(np.float32))
    mask_array = (slice3d > 0).astype(np.uint8)  # mask only non-zero areas
    print("Mask unique values:", np.unique(mask_array))  # Debug
    mask = sitk.GetImageFromArray(mask_array)

    # Set required 3D spatial properties
    image.SetSpacing((1.0, 1.0, 1.0))
    mask.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    mask.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    mask.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    return image, mask

# Extract radiomics features from 12 slices
def extract_radiomics_from_slices(image, extractor):
    slices = extract_12_slices(image)
    all_features = []

    for idx, slice2d in enumerate(slices):
        img2d, mask2d = convert_to_image_and_mask(slice2d)
        try:
            result = extractor.execute(img2d, mask2d)
            cleaned = {f"slice{idx}_{k}": v for k, v in result.items() if "diagnostics" not in k}
            all_features.append(cleaned)
        except Exception as e:
            print(f"Slice {idx} failed: {e}")
            return None

    merged = {}
    for d in all_features:
        merged.update(d)
    return merged

# Match filename in CSV with actual .nii.gz file path (search recursively)
def find_nii_file(scan_dir: Path, identifier: str):
    for file in scan_dir.rglob("*.nii.gz"):
        if identifier in file.name:
            return str(file)
    return None

# Batch Extraction Function
def extract_features_from_folder(scan_folder, label_csv):
    df_labels = pd.read_csv(label_csv)
    if 'Case' not in df_labels.columns:
        print("\u26a0\ufe0f 'Case' column not found — reloading with default column name.")
        df_labels = pd.read_csv(label_csv, header=None, names=['Case'])
    print("\u2705 CSV Columns:", df_labels.columns.tolist())

    rows = []
    scan_dir = Path(scan_folder)

    for _, row in df_labels.iterrows():
        identifier = row['Case'].replace('.nii.gz', '')
        label = 1  # Default label since it's not in the CSV

        nii_path = find_nii_file(scan_dir, identifier)
        if not nii_path:
            print(f"\u274c Missing: {identifier}")
            continue

        try:
            image = sitk.ReadImage(nii_path)
            features = extract_radiomics_from_slices(image, extractor)
            if features:
                features['label'] = label
                features['filename'] = Path(nii_path).name
                rows.append(features)
        except Exception as e:
            print(f"\u274c Error on {identifier}: {e}")

    return pd.DataFrame(rows)

# Set paths
scan_dir = "C:/Users/HenryLi/Downloads/Radiomic Model/Non evaluable"
label_csv = "C:/Users/HenryLi/Downloads/Radiomic Model/Radiomic Labels.csv"

# Run feature extraction
features_df = extract_features_from_folder(scan_dir, label_csv)

if not features_df.empty:
    features_df.to_csv("radiomics_features_fullvolume.csv", index=False)
    print(f"\u2705 Done! Extracted features from {len(features_df)} scans.")
else:
    print("\u26a0\ufe0f No features extracted. Check if file names match and are valid .nii.gz files.")

# Classifier Training
if not features_df.empty:
    df = features_df
    X = df.drop(columns=["label", "filename"])
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, preds))
    print("F1 Score:", f1_score(y_val, preds))
