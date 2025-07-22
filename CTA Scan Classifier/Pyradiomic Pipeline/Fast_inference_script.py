import SimpleITK as sitk
import numpy as np
import pandas as pd
import joblib
from radiomics import featureextractor
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed

# ---------- CONFIGURABLE EXTRACTOR (FAST VERSION) ----------
def get_fast_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('shape2D')
    # Uncomment if needed:
    # extractor.enableFeatureClassByName('glrlm')
    # extractor.enableFeatureClassByName('glszm')
    extractor.settings['label'] = 1
    extractor.settings['correctMask'] = False
    extractor.settings['force2D'] = True
    return extractor

# ---------- SLICE & MASK HANDLING ----------
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

    # Standard image metadata
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

def extract_features_from_image(image, extractor):
    slices = extract_12_slices(image)
    all_features = []
    for idx, slice2d in enumerate(slices):
        img, mask = convert_to_image_and_mask(slice2d)
        result = extractor.execute(img, mask)
        clean = {f"slice{idx}_{k}": v for k, v in result.items() if "diagnostics" not in k}
        all_features.append(clean)
    # Flatten all slice features into one row
    final = {}
    for f in all_features:
        final.update(f)
    return final

# ---------- PARALLEL WORKER ----------
def process_scan(scan_path, extractor, model, feature_names):
    try:
        image = sitk.ReadImage(str(scan_path))
        features = extract_features_from_image(image, extractor)
        X = pd.DataFrame([features])
        if feature_names is not None:
            X = X.reindex(columns=feature_names, fill_value=0)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]
        return {
            "filename": scan_path.name,
            "prediction": int(pred),
            "probability": round(prob, 4)
        }
    except Exception as e:
        print(f"❌ Failed on {scan_path.name}: {e}")
        return {
            "filename": scan_path.name,
            "prediction": None,
            "probability": None,
            "error": str(e)
        }

# ---------- BATCH PARALLEL INFERENCE ----------
def batch_predict_folder(scan_dir, model_path, output_csv="fast_inference_results.csv", n_jobs=-1):
    scan_dir = Path(scan_dir)
    scan_files = list(scan_dir.rglob("*.nii")) + list(scan_dir.rglob("*.nii.gz"))

    model = joblib.load(model_path)
    extractor = get_fast_extractor()
    feature_names = getattr(model, 'feature_names_in_', None)

    print(f"🧠 Loaded model. Starting parallel inference on {len(scan_files)} scans...")

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_scan)(scan_path, extractor, model, feature_names)
        for scan_path in tqdm(scan_files, desc="Processing scans")
    )

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Saved results to {output_csv}")


# ---------- EXAMPLE USAGE ----------
if __name__ == "__main__":
    batch_predict_folder(
        scan_dir=r"C:\Users\HenryLi\Downloads\Scans",
        model_path=r"C:\Users\HenryLi\Downloads\Radiomic Model\trained_fast_radiomics_model.pkl",
        output_csv="C:/Users/HenryLi/Downloads/Fast_scan_predictions.csv",
        n_jobs=-1  # Use all available cores
    )
