
import SimpleITK as sitk
import numpy as np
import pandas as pd
import joblib
from radiomics import featureextractor

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
    pred = model.predict(X)
    print(f"Predicted quality (1 = usable, 0 = not usable): {pred[0]}")
    return pred[0]

# Example usage:
# predict_scan_quality('path_to_new_scan.nii.gz', 'trained_radiomics_model.pkl')
