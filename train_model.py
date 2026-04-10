"""
train_model.py  —  A-DIDS XGBoost Training Pipeline
Trains, evaluates, and saves the IDS classifier using the ISOT Drone Dataset.

Usage:
    python3 train_model.py [--data PATH] [--output PATH]
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import (FEATURES, MODEL_PATH, DATA_PATH,
                           XGB_PARAMS, TEST_SIZE, RANDOM_STATE)

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="A-DIDS Training Pipeline")
parser.add_argument("--data",   default=DATA_PATH,  help="Input Parquet dataset")
parser.add_argument("--output", default=MODEL_PATH, help="Output model path (.pkl)")
args = parser.parse_args()

print("=" * 65)
print("  A-DIDS — XGBoost Training Pipeline")
print("=" * 65)

# ── 1. Load ───────────────────────────────────────────────────
print(f"\n[1/5] Loading dataset: {args.data}")
df = pd.read_parquet(args.data)
print(f"  Shape         : {df.shape}")
print(f"  Benign (0)    : {(df['label'] == 0).sum():,}")
print(f"  Attack (1)    : {(df['label'] == 1).sum():,}")
print(f"  Attack ratio  : {df['label'].mean():.2%}")

X = df[FEATURES]
y = df["label"]

# ── 2. Split ──────────────────────────────────────────────────
print(f"\n[2/5] Stratified 80/20 train-test split ...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train size : {X_train.shape[0]:,}")
print(f"  Test  size : {X_test.shape[0]:,}")

# ── 3. Train ──────────────────────────────────────────────────
print(f"\n[3/5] Training XGBoost ({XGB_PARAMS['n_estimators']} trees, "
      f"max_depth={XGB_PARAMS['max_depth']}) ...")
model = XGBClassifier(**XGB_PARAMS)
t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=40,
)
train_time = time.time() - t0
print(f"\n  Training completed in {train_time:.1f}s")

# ── 4. Evaluate ───────────────────────────────────────────────
print(f"\n[4/5] Evaluating on test set ...")
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
auc   = roc_auc_score(y_test, probs)

print("\n  Classification Report:")
print(classification_report(y_test, preds, target_names=["Benign", "Attack"]))

cm = confusion_matrix(y_test, preds)
print("  Confusion Matrix (rows=Actual, cols=Predicted):")
print(f"             Benign   Attack")
print(f"  Benign     {cm[0,0]:7,}  {cm[0,1]:7,}    (FP: {cm[0,1]:,})")
print(f"  Attack     {cm[1,0]:7,}  {cm[1,1]:7,}    (FN: {cm[1,0]:,})")
print(f"\n  ROC-AUC Score : {auc:.6f}")

print(f"\n  5-fold Cross-Validation (accuracy) ...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(
    XGBClassifier(**{**XGB_PARAMS, "n_estimators": 100, "verbosity": 0}),
    X, y, cv=cv, scoring="accuracy", n_jobs=-1
)
print(f"  Folds  : {np.round(cv_scores, 4)}")
print(f"  Mean   : {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")

# Feature importance
print(f"\n  Top-10 Feature Importances (gain):")
importances = pd.Series(model.get_booster().get_score(importance_type="gain"))
importances = importances.sort_values(ascending=False).head(10)
for feat, score in importances.items():
    print(f"    {feat:<20} {score:,.1f}")

# ── 5. Save ───────────────────────────────────────────────────
os.makedirs(os.path.dirname(args.output), exist_ok=True)
joblib.dump(model, args.output)
print(f"\n[5/5] Model saved to: {args.output}")
print(f"[DONE] Training complete.")
