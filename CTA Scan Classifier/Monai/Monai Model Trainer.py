import os
import numpy as np
import pydicom
import nibabel as nib
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from tqdm import tqdm
import matplotlib.pyplot as plt

from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Orientation,
    Spacing, Resize, RandRotate, RandFlip, RandZoom, RandGaussianNoise,
    RandAdjustContrast, RandAffine, EnsureType
)
from monai.networks.nets import DenseNet121, EfficientNetBN
from monai.data.meta_tensor import MetaTensor
from monai.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from collections import Counter

# Configuration
CONFIG = {
    "data_path": r"C:\Users\HenryLi\Downloads\Radiomic Model",
    "batch_size": 1,
    "num_workers": 0,
    "input_size": (128, 128, 128),
    "n_epochs": 20,
    "learning_rate": 1e-4,
    "pos_weight": 1.0  # Adjust based on class imbalance
}

# Preprocessing transform
preprocess = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(minv=0.0, maxv=1.0),
    Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    Orientation(axcodes="RAS"),
    Resize((128, 128, 128), mode="trilinear"),
    EnsureType()
])

def load_dicom_series(directory):
    dicom_files = [
        pydicom.dcmread(os.path.join(directory, f))
        for f in sorted(os.listdir(directory))
        if f.lower().endswith('.dcm')
    ]
    dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([f.pixel_array for f in dicom_files]).astype(np.float32)
    volume = np.transpose(volume, (1, 2, 0))  # (H, W, Depth)
    return MetaTensor(volume)  # NO channel metadata

# FIXED: Remove manual transpose
def load_nifti(filepath):
    img = nib.load(filepath)
    volume = img.get_fdata().astype(np.float32)
    return MetaTensor(volume)  # NO channel metadata

class CTADataset(Dataset):
    def __init__(self, file_list, labels, transforms=None):  # Removed is_dicom
        self.file_list = file_list
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        path = self.file_list[index]
        try:
            # Auto-detect file type
            if os.path.isdir(path):
                # For DICOM directories
                dicom_files = sorted([
                    os.path.join(path, f) 
                    for f in os.listdir(path) 
                    if f.lower().endswith('.dcm')
                ])
                # Pass DICOM directory to transform
                data = self.transforms(dicom_files) if self.transforms else None
            else:
                # For NIfTI files
                data = self.transforms(path) if self.transforms else None
                
            # Verify final shape
            if data is None or data.shape[0] != 1:
                raise ValueError(f"Bad channel dim: expected 1, got {data.shape[0] if data is not None else 'None'}")
                
            return data, torch.tensor(self.labels[index], dtype=torch.float)
            
        except Exception as e:
            print(f"❌ Error loading {path}: {str(e)}")
            # Return next valid sample
            return self[(index + 1) % len(self)]


augmentations = Compose([
    RandRotate(range_x=15, range_y=15, range_z=15, prob=0.5),
    RandFlip(prob=0.5),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    RandGaussianNoise(prob=0.3, std=0.01),
    RandAdjustContrast(gamma=(0.7, 1.5)),  # Gamma correction
    EnsureType()  # Convert to tensor
])

# 3. 3D CNN Model -------------------------------------------------------------
class QualityClassifier3D(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = monai.networks.nets.EfficientNetBN(
            "efficientnet-b0",
            spatial_dims=3,
            in_channels=1,
            num_classes=num_classes
        )
        
    def forward(self, x):
        # Return 1D tensor [batch_size]
        return torch.sigmoid(self.model(x)).squeeze(1)

# 4. Data Preparation ---------------------------------------------------------
def prepare_datasets(data_root):
    diagnostic_path = os.path.join(data_root, "Useable")
    non_diagnostic_path = os.path.join(data_root, "Non evaluable")

    file_list = []
    labels = []

    # Helper to find DICOM folders (contain at least one .dcm file)
    def find_dicom_dirs(root):
        return [d for d in glob.glob(os.path.join(root, "**/"), recursive=True)
                if any(f.endswith(".dcm") for f in os.listdir(d))]

    # Diagnostic
    diag_nii = glob.glob(os.path.join(diagnostic_path, "**", "*.nii*"), recursive=True)
    diag_dcm_dirs = find_dicom_dirs(diagnostic_path)
    file_list += diag_nii + diag_dcm_dirs
    labels += [1] * (len(diag_nii) + len(diag_dcm_dirs))

    # Non-Diagnostic
    non_diag_nii = glob.glob(os.path.join(non_diagnostic_path, "**", "*.nii*"), recursive=True)
    non_diag_dcm_dirs = find_dicom_dirs(non_diagnostic_path)
    file_list += non_diag_nii + non_diag_dcm_dirs
    labels += [0] * (len(non_diag_nii) + len(non_diag_dcm_dirs))

    # Split dataset
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_list, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.15, stratify=train_labels, random_state=42
    )

    # Create datasets
    train_ds = CTADataset(train_files, train_labels, transforms=Compose([preprocess, augmentations]))
    val_ds = CTADataset(val_files, val_labels, transforms=preprocess)
    test_ds = CTADataset(test_files, test_labels, transforms=preprocess)

    # Print dataset size and class distribution
    print(f"Training Set: {len(train_files)} scans")
    print("   Class balance:", Counter(train_labels))
    print(f"Validation Set: {len(val_files)} scans")
    print("   Class balance:", Counter(val_labels))
    print(f"Test Set: {len(test_files)} scans")
    print("   Class balance:", Counter(test_labels))

    return train_ds, val_ds, test_ds

# 5. Training and Evaluation --------------------------------------------------
def plot_metrics(history):
    epochs = range(1, len(history['val_loss']) + 1)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['val_loss'], 'b-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_acc'], 'g-', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_f1'], 'r-', label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QualityClassifier3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCELoss()
    pos_weight = torch.tensor([CONFIG["pos_weight"]]).to(device)

    train_ds, val_ds, test_ds = prepare_datasets(CONFIG["data_path"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"])

    history = {"val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(CONFIG["n_epochs"]):
        model.train()
        epoch_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['n_epochs']} [Training]")

        for batch_data in train_bar:
                # Skip empty or malformed samples
            if batch_data is None or batch_data[0] is None:
                continue
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device).float()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            pos_mask = (labels == 1)
            if pos_mask.any():
                loss += torch.mean(torch.abs(outputs[pos_mask] - labels[pos_mask])) * pos_weight

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['n_epochs']} [Validation]")

        with torch.no_grad():
            for val_data in val_bar:
                if val_data is None or val_data[0] is None:
                    continue
                inputs, labels = val_data[0].to(device), val_data[1].to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels.float())
                val_loss += loss.item()

                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_preds_bin = val_preds > 0.5
        val_auc = roc_auc_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds_bin)
        val_acc = np.mean(val_preds_bin == val_labels)

        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch+1}/{CONFIG['n_epochs']} | "
              f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

    plot_metrics(history)

    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"])
    return evaluate_model(model, test_loader, device)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for test_data in data_loader:
            inputs, labels = test_data[0].to(device), test_data[1].to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, all_preds)
    preds_bin = np.array(all_preds) > 0.5
    f1 = f1_score(all_labels, preds_bin)
    cm = confusion_matrix(all_labels, preds_bin)
    
    print("\nTest Results:")
    print(f"AUC: {auc:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return model

# 6. Run Pipeline -------------------------------------------------------------
if __name__ == "__main__":
    trained_model = train_model()
    torch.save(trained_model.state_dict(), "cta_monai_classifier.pt")
