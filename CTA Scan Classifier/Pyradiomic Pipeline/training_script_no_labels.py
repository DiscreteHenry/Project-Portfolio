# ---------- BoilerPlate ----------
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

# ---------- FEATURE EXTRACTOR ----------
def get_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('gldm')
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
            print(f" Slice {idx} failed: {e}")
            return None
    merged = {}
    for d in all_features:
        merged.update(d)
    return merged

# ---------- PARALLEL FEATURE EXTRACTOR FROM DIRECTORY ----------
def extract_features_from_folder_by_directory(scan_root_folder, output_csv, max_workers=6):
    scan_root = Path(scan_root_folder)
    usable_dir = scan_root / "usable"
    unusable_dir = scan_root / "unusable"

    extractor = get_extractor()
    write_lock = threading.Lock()
    first_row = True

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    def process_file(file_path, label):
        nonlocal first_row
        try:
            image = sitk.ReadImage(str(file_path))
            features = extract_radiomics_from_slices(image, extractor)
            if features:
                features['label'] = label
                features['filename'] = file_path.name
                with write_lock:
                    with open(output_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=features.keys())
                        if first_row:
                            writer.writeheader()
                            first_row = False
                        writer.writerow(features)
                return features
        except Exception as e:
            print(f"❌ Error on {file_path.name}: {e}")
        return None

    scan_paths = [(p, 1) for p in usable_dir.rglob("*.nii.gz")] + [(p, 0) for p in unusable_dir.rglob("*.nii.gz")]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, path, label) for path, label in scan_paths]
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Extracting features"):
            result = future.result()
            if result is not None:
                results.append(result)

    return pd.DataFrame(results)

# ---------- PATHS ----------
#This is where you can set the file path for your scans. This assumes the following folder structure. Change usable_dir and unusable_dir to suit your folder names.
#scan_dir/
#├── usable/     ← all usable scans (label = 1)
#│   ├── scan1.nii.gz
#│   ├── scan2.nii.gz
#│   └── ...
#├── unusable/   ← all unusable scans (label = 0)
#│   ├── scanA.nii.gz
#│   ├── scanB.nii.gz
#│   └── ...
scan_dir = "C:/Users/HenryLi/Downloads/Radiomic Model"
output_csv = "full_radiomics_features.csv"

# ---------- RUN FEATURE EXTRACTION ----------
features_df = extract_features_from_folder_by_directory(scan_dir, output_csv)

if not features_df.empty:
    print(f"Successfully Extracted features from {len(features_df)} scans.")
else:
    print("Warning. No features extracted. Check directory structure.")
    exit()

# ---------- TRAIN MODEL ----------
df = features_df
X = df.drop(columns=["label", "filename"])
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, "trained_radiomics_model.pkl")

# ---------- METRICS ----------
preds = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, preds))
print("F1 Score:", f1_score(y_val, preds))

cm = confusion_matrix(y_val, preds, labels=[1, 0])
TP, FN = cm[0]
FP, TN = cm[1]

print("\nConfusion Matrix:")
print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")

accuracy = (TP + TN) / (TP + TN + FP + FN)
recall_positive = TP / (TP + FN) if (TP + FN) > 0 else 0
recall_negative = TN / (TN + FP) if (TN + FP) > 0 else 0
balanced_accuracy = (recall_positive + recall_negative) / 2
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
f1 = 2 * (precision * recall_positive) / (precision + recall_positive) if (precision + recall_positive) > 0 else 0

print("\nDetailed Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"F1 Score (manual): {f1:.4f}")

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

pd.DataFrame(rows).to_csv("feature_importance_by_slice.csv", index=False)
print("Feature importance saved to 'feature_importance_by_slice.csv'")

# ---------- REORGANIZE FEATURE CSV ----------
def reorganize_feature_csv(input_file, output_file):
    from collections import defaultdict
    import csv

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        header = next(reader)

        base_feature_names = []
        for col in header:
            if col.startswith("slice0_"):
                base_feature_names.append(col.split("_", 1)[1])

        writer = csv.writer(outfile)
        writer.writerow(['Filename', 'Slice number', 'label'] + base_feature_names)

        for data_row in reader:
            label_value = data_row[-2]
            filename_value = data_row[-1]

            slice_groups = defaultdict(list)
            for idx, col_name in enumerate(header[:-2]):
                if '_' not in col_name:
                    continue
                parts = col_name.split('_', 1)
                if parts[0].startswith('slice'):
                    try:
                        slice_num = int(parts[0][5:])
                        feature_name = parts[1]
                        slice_groups[slice_num].append((feature_name, data_row[idx]))
                    except (ValueError, IndexError):
                        continue

            for slice_num in sorted(slice_groups.keys()):
                slice_dict = dict(slice_groups[slice_num])
                row = [
                    filename_value,
                    slice_num,
                    label_value,
                ]
                row.extend(slice_dict.get(feat, "") for feat in base_feature_names)
                writer.writerow(row)

    print(f"Reorganized CSV saved to: {output_file}")

reorganize_feature_csv("full_radiomics_features.csv", "reorganized_full_radiomics_features.csv")
