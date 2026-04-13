"""
train_multiclass.py  —  A-DIDS Multi-Class Attack Classifier
Trains an XGBoost model to categorize threats into 9 specific classes
(DoS, Spoofing, MITM, etc.) with real-time training progress.
"""

import os
import sys
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import DATA_PATH, FEATURES, CLASS_LABEL_COL, XGB_PARAMS, ATTACK_CLASSES, RANDOM_STATE
from modules.data_loader import A_DIDS_DataLoader

def train_multiclass():
    print("="*60)
    print("  A-DIDS: Multi-Class Attack Classification")
    print("="*60)

    # 1. Load Data
    loader = A_DIDS_DataLoader(DATA_PATH)
    X_train, X_test, y_train, y_test = loader.get_train_test_split(target_col=CLASS_LABEL_COL)
    
    # Create a small validation set for progress tracking
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
    )
    
    print(f"[Data] Training on {len(X_train_sub):,} samples, Validating on {len(X_val):,} samples.")

    # 2. Train Multi-Class XGBoost
    mc_params = XGB_PARAMS.copy()
    mc_params["objective"] = "multi:softprob"
    mc_params["num_class"] = 9
    mc_params["eval_metric"] = ["mlogloss"]
    
    model = XGBClassifier(**mc_params)
    
    print("\n[Training Start] Monitoring mlogloss progress...")
    model.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_train_sub, y_train_sub), (X_val, y_val)],
        verbose=True
    )

    # 3. Evaluate
    print("\n" + "─"*60)
    print("  EVALUATION RESULTS")
    print("─"*60)
    y_pred = model.predict(X_test)
    
    # Extract unique class names in order of their integer ID
    id_to_name = {}
    for name, cid in ATTACK_CLASSES.items():
        if cid not in id_to_name:
            id_to_name[cid] = name
        elif name == "Regular": # Prefer "Benign" or "Regular" for class 0
            id_to_name[cid] = "Benign"
            
    # Fallback to "Benign" for 0
    if 0 in id_to_name: id_to_name[0] = "Benign"
            
    unique_class_names = [id_to_name[i] for i in range(9)]
    
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=unique_class_names))

    # 4. Save Model
    out_path = "models/model_multiclass.pkl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    print(f"\n[Done] Multi-class model saved to {out_path}")

if __name__ == "__main__":
    train_multiclass()
