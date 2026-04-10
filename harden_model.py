"""
harden_model.py  —  Phase 6 (Adversarial Hardening)
Actively trains the XGBoost IDS to resist FGSM "Stealth Attacks".
Re-trains a hardened `model_hardened.pkl` and compares evasion rates.
"""

import os
import sys
import time

import joblib
import pandas as pd
from xgboost import XGBClassifier

from config.config import DATA_PATH, FEATURES, XGB_PARAMS
from modules.adversarial_engine import AdversarialEngine
from modules.data_loader import A_DIDS_DataLoader

def run_hardening():
    print("="*60)
    print("  Phase 6: Adversarial Hardening (FGSM)")
    print("="*60)
    
    # 1. Load Data
    loader = A_DIDS_DataLoader(DATA_PATH)
    X_train, X_test, y_train, y_test = loader.get_train_test_split()
    print(f"[Data] Loaded {len(X_train):,} training records.")

    # 2. Train Base Model
    print("\n[Stage 1] Training Base Model")
    base_model = XGBClassifier(**XGB_PARAMS)
    base_model.fit(X_train, y_train, verbose=False)
    
    # 3. Evaluate Base Vulnerability
    print("\n[Stage 2] Evaluating Vulnerability to FGSM")
    adv_engine = AdversarialEngine(base_model, epsilon=0.05)
    
    # Sample attacks to evaluate evasion
    attack_idx = (y_test == 1)
    X_attack_samples = X_test[attack_idx].head(500)
    
    # Convert dataframe to list of dicts for engine
    test_flows = X_attack_samples.to_dict(orient="records")
    base_results = adv_engine.evaluate_robustness(test_flows)
    print(f"  Base Evasion Rate: {base_results['evasion_rate']:.1%}")

    # 4. Generate Adversarial Training Data
    print("\n[Stage 3] Generating Adversarial Examples for Training")
    X_train_attack = X_train[y_train == 1].sample(frac=0.2, random_state=42)
    adv_train_rows = []
    
    # Generate stealth versions of 20% of the training attacks
    train_attack_flows = X_train_attack.to_dict(orient="records")
    for f in train_attack_flows:
        stealth_f = adv_engine.fgsm(f, epsilon=0.05, evasion=True)
        adv_train_rows.append(stealth_f)
    
    X_adv = pd.DataFrame(adv_train_rows)[FEATURES]
    y_adv = pd.Series([1] * len(X_adv))
    
    # Append to training set
    X_train_hardened = pd.concat([X_train, X_adv])
    y_train_hardened = pd.concat([y_train, y_adv])
    print(f"  Generated {len(X_adv):,} stealth samples. Retraining...")

    # 5. Train Hardened Model
    hardened_model = XGBClassifier(**XGB_PARAMS)
    hardened_model.fit(X_train_hardened, y_train_hardened, verbose=False)
    
    # 6. Evaluate Hardened Vulnerability
    print("\n[Stage 4] Evaluating Hardened Model")
    adv_engine_hard = AdversarialEngine(hardened_model, epsilon=0.05)
    hard_results = adv_engine_hard.evaluate_robustness(test_flows)
    
    print(f"\n  --- RESULTS ---")
    print(f"  Base Model Evasion Rate     : {base_results['evasion_rate']:.1%}")
    print(f"  Hardened Model Evasion Rate : {hard_results['evasion_rate']:.1%}")
    print(f"  Improvement                 : -{(base_results['evasion_rate'] - hard_results['evasion_rate']):.1%}")

    # Save
    out = "models/model_hardened.pkl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(hardened_model, out)
    print(f"\n[Done] Saved hardened model to {out}")

if __name__ == "__main__":
    run_hardening()
