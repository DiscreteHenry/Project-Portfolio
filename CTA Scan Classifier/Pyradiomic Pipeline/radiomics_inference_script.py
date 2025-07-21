import SimpleITK as sitk
import numpy as np
import pandas as pd
import joblib
from radiomics import featureextractor
import os
from pathlib import Path
from tqdm import tqdm

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

    image.SetSpacing((1.0, 1.0, 1.0))
    mask.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    mask.SetOrigin((0.0, 0.0, 0.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    mask.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return image, mask

def extract_features_from_image(image, extractor):
    slices = extract_12_slices(image)
    all_features = []
    for idx, slice2d in enumerate(slices):
        img, mask = convert_to_image_and_mask(slice2d)
        result = extractor.execute(img, mask)
        clean = {f"slice{idx}_{k}": v for k, v in result.items() if "diagnostics" not in k}
        all_features.append(clean)
    final = {}
    for f in all_features:
        final.update(f)
    return final

def predict_scan_quality(scan_path, model_path):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.settings['label'] = 1
    extractor.settings['correctMask'] = False
    extractor.settings['force2D'] = True

    image = sitk.ReadImage(scan_path)
    features = extract_features_from_image(image, extractor)
    model = joblib.load(model_path)

    X = pd.DataFrame([features])
    if hasattr(model, 'feature_names_in_'):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    pred = model.predict(X)
    prob = model.predict_proba(X)[0][1]
    print(f"Predicted quality: {pred[0]} (Probability diagnostic: {prob:.3f})")
    return pred[0], prob

def batch_predict_folder(scan_dir, model_path, output_csv="inference_results.csv"):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.settings['label'] = 1
    extractor.settings['correctMask'] = False
    extractor.settings['force2D'] = True

    model = joblib.load(model_path)
    results = []

    scan_dir = Path(scan_dir)
    scan_files = list(scan_dir.rglob("*.nii")) + list(scan_dir.rglob("*.nii.gz"))

    for scan_path in tqdm(scan_files, desc="Processing scans"):
        try:
            image = sitk.ReadImage(str(scan_path))
            features = extract_features_from_image(image, extractor)
            X = pd.DataFrame([features])
            if hasattr(model, 'feature_names_in_'):
                X = X.reindex(columns=model.feature_names_in_, fill_value=0)

            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0][1]
            results.append({
                "filename": scan_path.name,
                "prediction": int(pred),
                "probability": round(prob, 4)
            })
            print(f"✅ {scan_path.name}: pred={pred}, prob={prob:.3f}")
        except Exception as e:
            print(f"❌ Failed on {scan_path.name}: {e}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

# Example usage:
# Single scan prediction:
# predict_scan_quality(r"C:\Users\HenryLi\Downloads\Scans\example_scan.nii.gz", r"C:\Users\HenryLi\Downloads\Radiomic Model\trained_radiomics_model.pkl")

# Batch prediction for a folder:
batch_predict_folder(
    r"C:\Users\HenryLi\Downloads\Scans",
    r"C:\Users\HenryLi\Downloads\Radiomic Model\trained_radiomics_model.pkl",
    output_csv="C:/Users/HenryLi/Downloads/scan_predictions.csv"
)
