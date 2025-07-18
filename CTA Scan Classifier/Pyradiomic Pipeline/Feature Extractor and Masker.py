from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd
from pathlib import Path
import joblib
from tqdm import tqdm
import concurrent.futures
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Configure PyRadiomics
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()
extractor.settings['label'] = 1
extractor.settings['correctMask'] = False
extractor.settings['force2D'] = True

# Extract 12 representative slices from a 3D volume
def extract_12_slices(image):
    arr = sitk.GetArrayFromImage(image)
    z_slices = np.linspace(0, arr.shape[0] - 1, 10, dtype=int)
    axial = [arr[z, :, :] for z in z_slices]
    coronal = arr[arr.shape[0] // 2, :, :]
    sagittal = arr[:, :, arr.shape[2] // 2]
    return axial + [coronal, sagittal]

# Convert 2D slice to image and mask
def convert_to_image_and_mask(slice2d):
    slice3d = slice2d[np.newaxis, :, :]
    image = sitk.GetImageFromArray(slice3d.astype(np.float32))
    mask_array = (slice3d > 0).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_array)
    image.SetSpacing((1.0, 1.0, 1.0))
    mask.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    mask.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    mask.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image, mask

# Extract features from 12 slices
def extract_radiomics_from_slices(image, extractor):
    slices = extract_12_slices(image)
    all_features = []
    for idx, slice2d in enumerate(tqdm(slices, desc="  Processing slices", leave=False, unit="slice")):
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

# Find .nii.gz file by identifier
def find_nii_file(scan_dir: Path, identifier: str):
    for file in scan_dir.rglob("*.nii.gz"):
        if identifier in file.name:
            return str(file)
    return None

# Extract features from all scans in a folder
def extract_features_from_folder(scan_folder, label_csv):
    df_labels = pd.read_csv(label_csv)
    if not {'filename', 'label'}.issubset(df_labels.columns):
        raise ValueError("CSV must contain 'filename' and 'label' columns.")

    print("✅ CSV Columns:", df_labels.columns.tolist())
    scan_dir = Path(scan_folder)

    def process_row(row):
        identifier = row['filename'].replace('.nii.gz', '')
        raw_label = int(row['label'])
        label = 1 if raw_label == 4 else 0  # Normalize: 4 -> 1 (diagnostic), 1 -> 0 (non-diagnostic)
        nii_path = find_nii_file(scan_dir, identifier)
        if not nii_path:
            print(f"❌ Missing: {identifier}")
            return None
        try:
            image = sitk.ReadImage(nii_path)
            features = extract_radiomics_from_slices(image, extractor)
            if features:
                features['label'] = label
                features['filename'] = Path(nii_path).name
                return features
        except Exception as e:
            print(f"❌ Error on {identifier}: {e}")
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = list(tqdm(executor.map(process_row, [row for _, row in df_labels.iterrows()]), total=len(df_labels), desc="Extracting features", unit="scan"))

    rows = [f for f in futures if f is not None]
    return pd.DataFrame(rows)

# Set paths
scan_dir = "C:/Users/HenryLi/Downloads/Radiomic Model"
label_csv = "C:/Users/HenryLi/Downloads/Radiomic Model/Radiomic Labels.csv"

# Run feature extraction
features_df = extract_features_from_folder(scan_dir, label_csv)

if not features_df.empty:
    features_df.to_csv("radiomics_features_fullvolume.csv", index=False)
    print(f"✅ Done! Extracted features from {len(features_df)} scans.")
else:
    print("⚠️ No features extracted. Check if file names match and are valid .nii.gz files.")

# Classifier Training
if not features_df.empty:
    df = features_df
    X = df.drop(columns=["label", "filename"])
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "trained_radiomics_model.pkl")

    preds = clf.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, preds))
    print("F1 Score:", f1_score(y_val, preds))

    importances = pd.Series(clf.feature_importances_, index=X.columns)
    grouped = {}
    for feature, importance in importances.items():
        slice_key = feature.split('_')[0]
        grouped.setdefault(slice_key, []).append((feature, importance))

    rows = []
    for slice_name, feature_list in grouped.items():
        for fname, score in sorted(feature_list, key=lambda x: -x[1]):
            rows.append({"slice": slice_name, "feature": fname, "importance": score})

    pd.DataFrame(rows).to_csv("feature_importance_by_slice.csv", index=False)
    print("✅ Feature importance saved to 'feature_importance_by_slice.csv'")