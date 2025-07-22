import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from radiomics import featureextractor
import concurrent.futures
import csv
import threading
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---------- FAST FEATURE EXTRACTOR ----------
def get_fast_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('shape2D')
    extractor.settings['label'] = 1
    extractor.settings['correctMask'] = False
    extractor.settings['force2D'] = True
    return extractor

# ---------- SLICE HANDLING ----------
def extract_12_slices(image):
    arr = sitk.GetArrayFromImage(image)
    z_slices = np.linspace(0, arr.shape[0] - 1, 10, dtype=int)
    axial = [arr[z, :, :] for z in z_slices]
    coronal = arr[arr.shape[0] // 2, :, :]
    sagittal = arr[:, :, arr.shape[2] // 2]
    return axial + [coronal, sagittal]

def convert_to_image_and_mask(slice2d):
    slice3d = slice2d[np.newaxis, :, :]
    image = sitk.GetImageFromArray(slice3d.astype(np.float32))
    mask_array = (slice3d > 0).astype(np.uint8)
    mask = sitk.GetImageFromArray(mask_array)

    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    direction = (1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0)

    for obj in [image, mask]:
        obj.SetSpacing(spacing)
        obj.SetOrigin(origin)
        obj.SetDirection(direction)

    return image, mask

def extract_radiomics_from_slices(image, extractor):
    slices = extract_12_slices(image)
    all_features = []
    for idx, slice2d in enumerate(slices):
        img2d, mask2d = convert_to_image_and_mask(slice2d)
        try:
            result = extractor.execute(img2d, mask2d)
            clean = {f"slice{idx}_{k}": v for k, v in result.items() if "diagnostics" not in k}
            all_features.append(clean)
        except Exception as e:
            print(f"⚠️ Slice {idx} failed: {e}")
            return None
    merged = {}
    for d in all_features:
        merged.update(d)
    return merged

# ---------- SCAN MATCHING ----------
def find_nii_file(scan_dir: Path, identifier: str):
    for file in scan_dir.rglob("*.nii.gz"):
        if identifier in file.name:
            return str(file)
    return None

# ---------- PARALLEL FEATURE EXTRACTOR ----------
def extract_features_from_folder(scan_folder, label_csv, output_csv, max_workers=6):
    df_labels = pd.read_csv(label_csv)
    if not {'filename', 'label'}.issubset(df_labels.columns):
        raise ValueError("CSV must contain 'filename' and 'label' columns.")

    scan_dir = Path(scan_folder)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = get_fast_extractor()
    write_lock = threading.Lock()
    first_row = True

    # Clear existing CSV
    if output_path.exists():
        output_path.unlink()

    def process_row(row):
        nonlocal first_row
        identifier = row['filename'].replace('.nii.gz', '')
        raw_label = int(row['label'])
        label = 1 if raw_label == 4 else 0
        nii_path = find_nii_file(scan_dir, identifier)
        if not nii_path:
            print(f"❌ Missing scan: {identifier}")
            return None
        try:
            image = sitk.ReadImage(nii_path)
            features = extract_radiomics_from_slices(image, extractor)
            if features:
                features['label'] = label
                features['filename'] = Path(nii_path).name
                with write_lock:
                    with open(output_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=features.keys())
                        if first_row:
                            writer.writeheader()
                            first_row = False
                        writer.writerow(features)
                return features
        except Exception as e:
            print(f"❌ Error on {identifier}: {e}")
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_row, row) for _, row in df_labels.iterrows()]
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting features"):
            result = future.result()
            if result is not None:
                results.append(result)

    return pd.DataFrame(results)

# ---------- PATHS ----------
scan_dir = "C:/Users/HenryLi/Downloads/Radiomic Model"
label_csv = "C:/Users/HenryLi/Downloads/Radiomic Model/Radiomic Labels.csv"
output_csv = "radiomics_features_fast.csv"

# ---------- RUN FEATURE EXTRACTION ----------
features_df = extract_features_from_folder(scan_dir, label_csv, output_csv)

if not features_df.empty:
    print(f"✅ Extracted features from {len(features_df)} scans.")
else:
    print("⚠️ No features extracted. Check file names and CSV.")
    exit()

# ---------- TRAIN MODEL ----------
df = features_df
X = df.drop(columns=["label", "filename"])
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, "trained_fast_radiomics_model.pkl")

# ---------- METRICS ----------
preds = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, preds))
print("F1 Score:", f1_score(y_val, preds))

#adding in confusion matrix
cm = confusion_matrix(y_val, preds, labels=[1, 0]) #the 1 is first to ensure positive class is first
TP, FN = cm[0]
FP, TN = cm[1]

print("\nConfusion Matrix:")
print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
#Calculate metrics
accuracy = (TP + TN) / (TP + TN + FP + FN)
recall_positive = TP / (TP + FN) if (TP + FN) > 0 else 0
recall_negative = TN / (TN + FP) if (TN + FP) > 0 else 0
balanced_accuracy = (recall_positive + recall_negative) / 2
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
f1 = 2 * (precision * recall_positive) / (precision + recall_positive) if (precision + recall_positive) > 0 else 0

print("\nDetailed Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"F1 Score: (manual): {f1:.4f}")
# ---------- FEATURE IMPORTANCE ----------
importances = pd.Series(clf.feature_importances_, index=X.columns)
grouped = {}
for feature, importance in importances.items():
    slice_key = feature.split('_')[0]
    grouped.setdefault(slice_key, []).append((feature, importance))

rows = []
for slice_name, feature_list in grouped.items():
    for fname, score in sorted(feature_list, key=lambda x: -x[1]):
        rows.append({"slice": slice_name, "feature": fname, "importance": score})

pd.DataFrame(rows).to_csv("fast_feature_importance_by_slice.csv", index=False)
print("✅ Feature importance saved to 'fast_feature_importance_by_slice.csv'")
