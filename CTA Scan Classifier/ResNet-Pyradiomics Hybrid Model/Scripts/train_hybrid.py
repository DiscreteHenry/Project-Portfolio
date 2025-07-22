# scripts/train_hybrid.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

radiomics_df = pd.read_csv("../data/radiomics_features.csv")
deep_df = pd.read_csv("../data/deep_features.csv")

# Merge on filename
merged = pd.merge(radiomics_df, deep_df, on=["filename", "label"], how="inner")
print(f"✅ Merged dataset size: {merged.shape}")

X = merged.drop(columns=["filename", "label"])
y = merged["label"]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train hybrid model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
joblib.dump(clf, "../model/hybrid_rf_model.pkl")

# Evaluate
preds = clf.predict(X_val)
acc = accuracy_score(y_val, preds)
f1 = f1_score(y_val, preds)
print(f"\n📊 Hybrid Model Evaluation:\nAccuracy: {acc:.4f}\nF1 Score: {f1:.4f}")
