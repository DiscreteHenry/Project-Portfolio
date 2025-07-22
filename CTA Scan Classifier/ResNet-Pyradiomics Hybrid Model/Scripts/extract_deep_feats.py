# scripts/extract_deep_feats.py
import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
import nibabel as nib
import pydicom
import cv2
import pathlib
from tqdm import tqdm
from model.slice_resnet import CTAQuality2D, extract_12_slices, normalize_volume, load_scan, find_file_recursive

class DeepFeatureDataset(Dataset):
    def __init__(self, labels_csv, root_dir, model_path):
        self.data = pd.read_csv(labels_csv)
        self.root_dir = root_dir
        self.model = CTAQuality2D()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['filename']
        label = 1 if int(row['label']) >= 3 else 0
        path = find_file_recursive(self.root_dir, filename)
        vol = load_scan(path)
        vol = normalize_volume(vol)
        slices = extract_12_slices(vol)
        slices = torch.tensor(slices).unsqueeze(1).float().to(self.device)
        with torch.no_grad():
            B, S, C, H, W = slices.unsqueeze(0).shape
            flat = slices.view(S, C, H, W)
            feats = self.model.slice_cnn(flat)
            pooled = feats.view(1, S, -1).mean(dim=1).squeeze().cpu().numpy()
        return pooled, label, filename

if __name__ == '__main__':
    csv_path = "../data/Radiomic Labels.csv"
    scan_root = r"C:\Users\HenryLi\Downloads\Radiomic Model"
    model_path = "../model/best_resnetgoodmodel.pt"

    dataset = DeepFeatureDataset(csv_path, scan_root, model_path)
    all_feats = []
    print("\n🔄 Extracting deep features with progress bar:")
    for i in tqdm(range(len(dataset)), desc="Extracting", unit="scan"):
        feats, label, fname = dataset[i]
        record = dict(zip([f"deep_feat_{j}" for j in range(len(feats))], feats))
        record.update({"label": label, "filename": fname})
        all_feats.append(record)
    df = pd.DataFrame(all_feats)
    df.to_csv("../data/deep_features.csv", index=False)
    print("✅ Deep features saved to ../data/deep_features.csv")
