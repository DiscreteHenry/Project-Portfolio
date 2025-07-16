import os
import numpy as np
import pydicom
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity,
    Spacing, Resize, RandRotate, RandFlip, RandZoom, RandGaussianNoise,
    RandAdjustContrast, RandAffine, EnsureType
)
from monai.networks.nets import DenseNet121, EfficientNetBN
from monai.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# Configuration
CONFIG = {
    "data_path": "/path/to/your/dataset",
    "batch_size": 4,
    "num_workers": 4,
    "input_size": (128, 128, 128),
    "n_epochs": 20,
    "learning_rate": 1e-4,
    "pos_weight": 4.0  # Adjust based on class imbalance
}

# 1. Data Loading and Preprocessing -------------------------------------------
def load_dicom_series(directory):
    """Load DICOM series into 3D numpy array"""
    files = [pydicom.dcmread(os.path.join(directory, f)) 
             for f in sorted(os.listdir(directory)) 
             if f.endswith('.dcm')]
    files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([f.pixel_array for f in files])
    return volume.astype(np.float32), files[0]

def load_nifti(filepath):
    """Load NIfTI file into numpy array"""
    img = nib.load(filepath)
    return img.get_fdata().astype(np.float32), img.affine

class CTADataset(Dataset):
    def __init__(self, file_list, labels, transforms=None, is_dicom=True):
        self.file_list = file_list
        self.labels = labels
        self.transforms = transforms
        self.is_dicom = is_dicom  # True for DICOM, False for NIfTI
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # Load data based on format
        if self.is_dicom:
            data, _ = load_dicom_series(self.file_list[index])
        else:
            data, _ = load_nifti(self.file_list[index])
            
        # Apply preprocessing transforms
        if self.transforms:
            data = self.transforms(data)
            
        label = self.labels[index]
        return data, label

# 2. Preprocessing Pipeline ---------------------------------------------------
preprocess = Compose([
    EnsureChannelFirst(),        # Add channel dimension: (H,W,D) -> (C,H,W,D)
    ScaleIntensity(minv=0, maxv=1),  # Normalize to [0, 1]
    Spacing(pixdim=(1.0, 1.0, 1.0), mode='bilinear'),  # Resample to isotropic
    Resize(CONFIG["input_size"], mode='trilinear'),  # Resize to uniform dimensions
])

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
        # Alternative: DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
        
    def forward(self, x):
        return torch.sigmoid(self.model(x)).squeeze(1)

# 4. Data Preparation ---------------------------------------------------------
def prepare_datasets(data_root):
    # This should be customized to your directory structure
    # Example structure:
    # data_root/
    #   ├── diagnostic/
    #   │   ├── scan1/ (DICOM series)
    #   │   ├── scan2.nii.gz
    #   │   └── ...
    #   └── non_diagnostic/
    #       ├── scan101/
    #       └── ...
    
    diagnostic_path = os.path.join(data_root, "diagnostic")
    non_diagnostic_path = os.path.join(data_root, "non_diagnostic")
    
    file_list = []
    labels = []
    
    # Load DICOM series (directories)
    for scan_dir in os.listdir(diagnostic_path):
        full_path = os.path.join(diagnostic_path, scan_dir)
        if os.path.isdir(full_path):
            file_list.append(full_path)
            labels.append(1)  # Diagnostic quality
    
    # Load NIfTI files
    for nifti_file in os.listdir(diagnostic_path):
        if nifti_file.endswith(('.nii', '.nii.gz')):
            file_list.append(os.path.join(diagnostic_path, nifti_file))
            labels.append(1)
    
    # Repeat for non-diagnostic
    for scan_dir in os.listdir(non_diagnostic_path):
        full_path = os.path.join(non_diagnostic_path, scan_dir)
        if os.path.isdir(full_path):
            file_list.append(full_path)
            labels.append(0)  # Non-diagnostic
    
    for nifti_file in os.listdir(non_diagnostic_path):
        if nifti_file.endswith(('.nii', '.nii.gz')):
            file_list.append(os.path.join(non_diagnostic_path, nifti_file))
            labels.append(0)
    
    # Split dataset
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_list, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.15, stratify=train_labels, random_state=42
    )
    
    # Create datasets
    train_ds = CTADataset(
        train_files, 
        train_labels,
        transforms=Compose([preprocess, augmentations]),
        is_dicom=isinstance(train_files[0], str)  # Auto-detect format
    )
    
    val_ds = CTADataset(
        val_files, 
        val_labels,
        transforms=preprocess,
        is_dicom=isinstance(val_files[0], str)
    )
    
    test_ds = CTADataset(
        test_files, 
        test_labels,
        transforms=preprocess,
        is_dicom=isinstance(test_files[0], str)
    )
    
    return train_ds, val_ds, test_ds

# 5. Training and Evaluation --------------------------------------------------
def train_model():
    # Initialize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QualityClassifier3D().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCELoss()
    pos_weight = torch.tensor([CONFIG["pos_weight"]]).to(device)
    
    # Load data
    train_ds, val_ds, test_ds = prepare_datasets(CONFIG["data_path"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], 
                             shuffle=True, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], 
                           num_workers=CONFIG["num_workers"])
    
    # Training loop
    for epoch in range(CONFIG["n_epochs"]):
        model.train()
        epoch_loss = 0
        
        for batch_data in train_loader:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            
            # Apply class weighting
            pos_mask = (labels == 1)
            if pos_mask.any():
                loss += torch.mean(torch.abs(outputs[pos_mask] - labels[pos_mask])) * pos_weight
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for val_data in val_loader:
                inputs, labels = val_data[0].to(device), val_data[1].to(device)
                outputs = model(inputs)
                
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        val_auc = roc_auc_score(val_labels, val_preds)
        val_preds_bin = np.array(val_preds) > 0.5
        val_f1 = f1_score(val_labels, val_preds_bin)
        
        print(f"Epoch {epoch+1}/{CONFIG['n_epochs']} | "
              f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
    
    # Final evaluation on test set
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], 
                            num_workers=CONFIG["num_workers"])
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
    torch.save(trained_model.state_dict(), "cta_quality_classifier.pth")
