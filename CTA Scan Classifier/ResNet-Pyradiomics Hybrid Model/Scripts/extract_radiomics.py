# scripts/extract_radiomics.py
import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from radiomics import featureextractor
import concurrent.futures
import csv
import threading
from tqdm import tqdm

def get_fast_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    for cls in ['firstorder', 'glrlm', 'glszm', 'glcm', 'ngtdm', 'shape2D']:
        extractor.enableFeatureClassByName(cls)
    extractor.settings.update({"label": 1, "correctMask": False, "force2D": True})
    return extractor

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
    for obj in [image, mask]:
        obj.SetSpacing((1.0, 1.0, 1.0))
        obj.SetOrigin((0.0, 0.0, 0.0))
        obj.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
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
            print(f"Slice {idx} failed: {e}")
            return None
    merged = {}
    for d in all_features:
        merged.update(d)
    return merged

def find_nii_file(scan_dir: Path, identifier: str):
    for file in scan_dir.rglob("*.nii.gz"):
        if identifier in file.name:
            return str(file)
    return None

def extract_features_from_folder(scan_folder, label_csv, output_csv, max_workers=6):
    df_labels = pd.read_csv(label_csv)
    scan_dir = Path(scan_folder)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extractor = get_fast_extractor()
    write_lock = threading.Lock()
    first_row = True
    if output_path.exists():
        output_path.unlink()

    def process_row(row):
        nonlocal first_row
        identifier = row['filename'].replace('.nii.gz', '')
        label = 1 if int(row['label']) == 4 else 0
        nii_path = find_nii_file(scan_dir, identifier)
        if not nii_path:
            print(f"Missing scan: {identifier}")
            return None
        try:
            image = sitk.ReadImage(nii_path)
            features = extract_radiomics_from_slices(image, extractor)
            if features:
                features.update({'label': label, 'filename': Path(nii_path).name})
                with write_lock:
                    with open(output_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=features.keys())
                        if first_row:
                            writer.writeheader()
                            first_row = False
                        writer.writerow(features)
                return features
        except Exception as e:
            print(f"Error on {identifier}: {e}")
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_row, df_labels.to_dict('records')), total=len(df_labels)))

if __name__ == '__main__':
    scan_dir = r"C:\Users\HenryLi\Downloads\Radiomic Model"
    label_csv = "../data/Radiomic Labels.csv"
    output_csv = "../data/radiomics_features.csv"
    extract_features_from_folder(scan_dir, label_csv, output_csv)
