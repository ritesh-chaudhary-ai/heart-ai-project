"""
CardioAI Pro — Model Training Script
XGBoost with cross-validation + joblib export

Usage:
    python train_model.py

Expects heart.csv in the same directory.
Produces heart_app.pkl consumed by app.py.

Required CSV columns (in any order):
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal, target
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# REQUIRED FEATURE ORDER — must match app.py sidebar inputs exactly
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]
TARGET_COL = "target"

# ─────────────────────────────────────────────────────────────────────────────
# Try XGBoost, fall back to Random Forest
# ─────────────────────────────────────────────────────────────────────────────
try:
    import xgboost as xgb
    USE_XGB = True
    print("✅ XGBoost found — using XGBClassifier")
    print(f"   XGBoost version: {xgb.__version__}")
except ImportError:
    USE_XGB = False
    print("⚠️  XGBoost not installed — falling back to RandomForest")
    print("   Install with: pip install xgboost")

# ─────────────────────────────────────────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n📂 Loading heart.csv ...")
try:
    df = pd.read_csv("heart.csv")
except FileNotFoundError:
    print("❌  heart.csv not found. Place it in the same directory as this script.")
    sys.exit(1)

print(f"   Shape: {df.shape}")

# ── Validate all required columns are present ────────────────────────────────
all_required = REQUIRED_FEATURES + [TARGET_COL]
missing_cols = [c for c in all_required if c not in df.columns]
if missing_cols:
    print(f"\n❌  Missing columns in heart.csv: {missing_cols}")
    print(f"   Found columns: {list(df.columns)}")
    sys.exit(1)

# ── Drop any unexpected extra columns so column order is deterministic ────────
df = df[all_required]

# ── Basic dataset info ────────────────────────────────────────────────────────
pos = df[TARGET_COL].sum()
print(f"   Positive cases (High Risk): {pos} / {len(df)}  ({pos / len(df) * 100:.1f}%)")

# ── Check for missing values ──────────────────────────────────────────────────
null_counts = df.isnull().sum()
if null_counts.any():
    print(f"\n⚠️  Missing values detected — filling with column median:")
    print(null_counts[null_counts > 0])
    df = df.fillna(df.median(numeric_only=True))

# ─────────────────────────────────────────────────────────────────────────────
# Features / target
# ─────────────────────────────────────────────────────────────────────────────
X = df[REQUIRED_FEATURES].copy()
y = df[TARGET_COL].copy()
feature_names = REQUIRED_FEATURES  # locked order, matches app.py

# ─────────────────────────────────────────────────────────────────────────────
# Train / test split (stratified)
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n   Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

# ─────────────────────────────────────────────────────────────────────────────
# Scale
# ─────────────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# Build model
# ─────────────────────────────────────────────────────────────────────────────
if USE_XGB:
    from xgboost import XGBClassifier

    # Detect XGBoost version to handle deprecated params safely
    xgb_major = int(xgb.__version__.split(".")[0])

    xgb_params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",   # kept in constructor (works ≥1.6)
        random_state=42,
        n_jobs=-1,
    )
    # use_label_encoder was removed in XGBoost 1.6+ — only pass it on older versions
    if xgb_major < 2:
        xgb_params["use_label_encoder"] = False

    model = XGBClassifier(**xgb_params)
    model_type = "XGBoost"

else:
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    model_type = "RandomForest"

print(f"\n🤖 Model: {model_type}")

# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold Stratified Cross-Validation (on training set only)
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔁 5-Fold Stratified Cross-Validation ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring="roc_auc")
print(f"   ROC-AUC per fold : {[f'{s:.4f}' for s in cv_scores]}")
print(f"   Mean ± Std       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Final fit on full training set
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏋️  Fitting final model on full training set ...")
model.fit(X_train_s, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on held-out test set
# ─────────────────────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n📊 Test Set Results")
print(f"   Accuracy : {acc * 100:.2f}%")
print(f"   ROC-AUC  : {auc:.4f}")

print(f"\n{classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'])}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("   Confusion Matrix:")
print(f"   TN={cm[0,0]}  FP={cm[0,1]}")
print(f"   FN={cm[1,0]}  TP={cm[1,1]}")

# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────
importances = model.feature_importances_
fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print("\n📌 Feature Importances (descending):")
print(f"   {'Feature':<14} {'Importance':>10}   Bar")
print(f"   {'-'*14} {'-'*10}   ---")
for feat, imp in fi.items():
    bar = "█" * max(1, int(imp * 40))
    print(f"   {feat:<14} {imp:>10.4f}   {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# Validate payload keys match what app.py expects
# ─────────────────────────────────────────────────────────────────────────────
# app.py does:  data["model"], data["scaler"], data["columns"]
# Confirm column order in scaler matches REQUIRED_FEATURES
assert list(fi.index) != [], "Feature importances must not be empty"
assert len(feature_names) == len(REQUIRED_FEATURES), "Feature count mismatch"
assert feature_names == REQUIRED_FEATURES, (
    f"Feature order mismatch!\n  Got     : {feature_names}\n  Expected: {REQUIRED_FEATURES}"
)

payload = {
    # ── Keys consumed by app.py (do not rename) ──
    "model":   model,
    "scaler":  scaler,
    "columns": feature_names,          # list[str] in REQUIRED_FEATURES order

    # ── Metadata (informational only) ────────────
    "model_type":          model_type,
    "test_accuracy":       round(acc, 4),
    "test_auc":            round(auc, 4),
    "cv_auc_mean":         round(float(cv_scores.mean()), 4),
    "cv_auc_std":          round(float(cv_scores.std()), 4),
    "feature_importances": fi.to_dict(),
    "train_samples":       len(X_train),
    "test_samples":        len(X_test),
}

# ─────────────────────────────────────────────────────────────────────────────
# Save artefact
# ─────────────────────────────────────────────────────────────────────────────
output_path = "heart_app.pkl"
joblib.dump(payload, output_path)

print(f"\n✅ Model saved → {output_path}")
print(f"   Model type : {model_type}")
print(f"   Accuracy   : {acc * 100:.2f}%")
print(f"   ROC-AUC    : {auc:.4f}")
print(f"   CV AUC     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\n🚀 Run the app with:  streamlit run app.py")