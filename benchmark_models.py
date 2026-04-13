"""
benchmark_models.py  —  Professional Model Comparison Suite
Evaluates multiple supervised learning algorithms side-by-side to justify
architectural choices (XGBoost vs. Random Forest vs. Logistic Regression).
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import DATA_PATH, FEATURES, LABEL_COL, RANDOM_STATE
from modules.data_loader import A_DIDS_DataLoader

def run_individual_benchmark(name, model, X_train, X_test, y_train, y_test):
    print(f"\n" + "─"*60)
    print(f"  ALGORITHM: {name}")
    print("─"*60)
    
    # 1. Training
    print(f"  [1/3] Training {name} on {len(X_train):,} rows...")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"        (✓) Training complete in {train_time:.2f}s")
    
    # 2. Inference & Metrics
    print(f"  [2/3] Evaluating on {len(X_test):,} unseen samples...")
    start_infer = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    infer_time = time.time() - start_infer
    ms_per_packet = (infer_time / len(X_test)) * 1000
    
    # 3. Report
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n[RESULTS: {name}]")
    print(f"  Accuracy  : {acc:.4%}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  Latency   : {ms_per_packet:.4f} ms per packet")
    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"]))
    
    return {
        "Model": name,
        "Accuracy": f"{acc:.2%}",
        "ROC-AUC": f"{auc:.4f}",
        "Train Time": f"{train_time:.2f}s",
        "Latency": f"{ms_per_packet:.4f}ms"
    }

def main():
    print("="*60)
    print("  A-DIDS: Professional Model Benchmarking Suite")
    print("  Objective: Evaluate Performance vs. Latency Trade-offs")
    print("="*60)

    # Load Data
    loader = A_DIDS_DataLoader(DATA_PATH)
    # Using 10% sample for stability in benchmarking environment (294k rows)
    X_train, X_test, y_train, y_test = loader.get_train_test_split(test_size=0.1)
    
    # Stratified downsample to 10% for the benchmark to keep it fast/stable
    X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=RANDOM_STATE, stratify=y_test)
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=RANDOM_STATE, stratify=y_train)
    
    print(f"\n[SETUP] Using statistically significant 10% sample: {len(X_train):,} train, {len(X_test):,} test")

    models_to_test = [
        ("XGBoost", XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]

    summary_stats = []

    for name, model in models_to_test:
        stats = run_individual_benchmark(name, model, X_train, X_test, y_train, y_test)
        summary_stats.append(stats)

    # Save summary table for README
    os.makedirs("results", exist_ok=True)
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_markdown("results/MODEL_BENCHMARK.md", index=False)
    
    print("\n" + "="*60)
    print("  BENCHMARK COMPLETE")
    print("  Final comparison table saved to results/MODEL_BENCHMARK.md")
    print("="*60)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
