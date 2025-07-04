import os
import numpy as np
import torch
import nibabel as nib
import cv2
import pydicom

# ----------------------------
# Model Definition
# ----------------------------
import torch.nn as nn

class SliceCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super(SliceCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CTAQuality2D(nn.Module):
    def __init__(self, feature_dim=128):
        super(CTAQuality2D, self).__init__()
        self.slice_cnn = SliceCNN(feature_dim)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.slice_cnn(x)
        feats = feats.view(B, S, -1)
        pooled = feats.mean(dim=1)
        return self.regressor(pooled).squeeze(1)

# ----------------------------
# Preprocessing Functions
# ----------------------------
import pathlib

def load_scan(file_path):
    file_path = str(pathlib.Path(file_path))
    if file_path.endswith('.nii') or file_path.endswith('.nii.gz'):
        img = nib.load(file_path)
        return img.get_fdata()
    elif os.path.isdir(file_path):
        slices = [pydicom.dcmread(os.path.join(file_path, f)) for f in os.listdir(file_path) if f.endswith('.dcm')]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        volume = np.stack([s.pixel_array for s in slices], axis=-1)
        return volume
    else:
        raise ValueError(f"Unsupported format or bad path: {file_path}")

def normalize_volume(volume):
    volume = volume.astype(np.float32)
    volume -= np.min(volume)
    volume /= np.max(volume)
    return volume

def extract_12_slices(volume):
    slices = []
    z_len = volume.shape[2]
    axial_indices = np.linspace(0, z_len - 1, 10, dtype=int)
    axial_slices = [volume[:, :, idx] for idx in axial_indices]

    coronal_idx = volume.shape[1] // 2
    coronal_slice = volume[:, coronal_idx, :]

    sagittal_idx = volume.shape[0] // 2
    sagittal_slice = volume[sagittal_idx, :, :]

    slices.extend(axial_slices)
    slices.append(coronal_slice)
    slices.append(sagittal_slice)
    return slices

def preprocess_scan(path, target_size=(224, 224)):
    volume = load_scan(path)
    volume = normalize_volume(volume)
    slices = extract_12_slices(volume)
    resized = [cv2.resize(s, target_size, interpolation=cv2.INTER_AREA) for s in slices]
    return np.stack(resized, axis=0)

# ----------------------------
# Prediction Function
# ----------------------------
def predict_quality(model_path, scan_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CTAQuality2D()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    slices = preprocess_scan(scan_path)
    slices = torch.tensor(slices).unsqueeze(1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        prediction = model(slices)

    return prediction.item()

# ----------------------------
# Example Usage
# ----------------------------
if __name__ == "__main__":
    scan_path = r"C:\\Users\\HenryLi\\Downloads\\Scans\\11047\\20\\PREFFIR-11047 (CorCTAAdapt  0.75  B26f  75%).nii.gz"
    model_path = "best_model.pt"
    score = predict_quality(model_path, scan_path)
    print(f"Predicted Quality Score: {score:.2f}")
