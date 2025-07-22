# scripts/infer_hybrid.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report

# Load model and features
model = joblib.load("../model/hybrid_rf_model.pkl")
deep_df = pd.read_csv("../data/deep_features.csv")
radiomics_df = pd.read_csv("../data/radiomics_features.csv")

# Merge on filename and label
merged = pd.merge(radiomics_df, deep_df, on=["filename", "label"], how="inner")
X = merged.drop(columns=["filename", "label"])
y = merged["label"]

# Predict and report
preds = model.predict(X)
print("\n🧪 Inference Results:")
print(classification_report(y, preds, digits=4))
