#Feature Extractor
from radiomics import featureextractor
import SimpleITK as sitk
import numpy as np
import os
import pandas as pd

# Configure PyRadiomics
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()
extractor.disableAllImageTypes()  # Optional: disable wavelets, log-sigma, etc.
extractor.enableImageTypeByName("original")

#Auto Masker
def create_full_mask(image):
    array = sitk.GetArrayFromImage(image)
    mask_array = np.ones_like(array, dtype=np.uint8)

    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)
    return mask

#Batch Extraction Folder
def extract_features_from_folder(scan_folder, label_csv):
    df_labels = pd.read_csv(label_csv)
    rows = []

    for _, row in df_labels.iterrows():
        filename = row['filename']
        label = 1 if row['label'] >= 3 else 0

        scan_path = os.path.join(scan_folder, filename)
        if not os.path.exists(scan_path):
            print(f"Missing: {scan_path}")
            continue

        try:
            image = sitk.ReadImage(scan_path)
            mask = create_full_mask(image)

            result = extractor.execute(image, mask)
            clean_result = {
                k: v for k, v in result.items()
                if "diagnostics" not in k
            }

            clean_result['label'] = label
            clean_result['filename'] = filename
            rows.append(clean_result)
        except Exception as e:
            print(f"Error on {filename}: {e}")

    return pd.DataFrame(rows)

#4. Run the pipeline and save features
scan_dir = "C:/Users/HenryLi/Downloads/ImageCAS Model"
label_csv = "C:/Users/HenryLi/Downloads/ImageCAS Model/ImageCAS Labels.csv"

features_df = extract_features_from_folder(scan_dir, label_csv)
features_df.to_csv("radiomics_features_fullvolume.csv", index=False)

print("Done! Extracted features from", len(features_df), "scans.")

#5 Classifier Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv("radiomics_features_fullvolume.csv")
X = df.drop(columns=["label", "filename"])
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

preds = clf.predict(X_val)
print("Accuracy:", accuracy_score(y_val, preds))
print("F1 Score:", f1_score(y_val, preds))