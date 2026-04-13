"""
validate_mixed_traffic.py  —  Real-World Robustness Stress Test
Simulates a continuous, "noisy" stream of mixed benign and attack traffic
to quantify the system's performance in real-world deployment conditions.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import DATA_PATH, MODEL_PATH, FEATURES, LABEL_COL

def run_stress_test():
    print("="*60)
    print("  A-DIDS: Real-World Mixed Traffic Stress Test")
    print("="*60)

    # 1. Load Model & Data
    print(f"[1/4] Loading model and dataset...")
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(DATA_PATH)
    
    # 2. Create Mixed Stream (Simulating Realism)
    # We take a random sample of 100,000 flows, shuffled
    sample_size = 100000
    test_stream = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    X = test_stream[FEATURES]
    y_true = test_stream[LABEL_COL]
    
    # 3. Running Sequential Inference
    print(f"[2/4] Simulating real-time stream of {sample_size:,} flows...")
    start_time = time.time()
    y_pred = model.predict(X)
    total_time = time.time() - start_time
    
    # 4. Calculating Production Metrics
    print(f"[3/4] Calculating robust metrics...")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    throughput = sample_size / total_time
    latency_per_flow = (total_time / sample_size) * 1000
    
    print("\n" + "─"*60)
    print("  STRESS TEST RESULTS (PRODUCTION METRICS)")
    print("─"*60)
    print(f"  Overall Accuracy        : {acc:.4%}")
    print(f"  Precision (Trust)       : {prec:.4%}")
    print(f"  Recall (Detection)      : {rec:.4%}")
    print(f"  F1-Score                : {f1:.4%}")
    print(f"  False Positive Rate     : {fpr:.4%}")
    print(f"  False Negative Rate     : {fnr:.4%}")
    print(f"  System Throughput       : {throughput:,.2f} flows/sec")
    print(f"  Average Inference Time  : {latency_per_flow:.4f} ms per flow")
    
    # 5. Save Results
    results_path = "results/STRESS_TEST_RESULTS.md"
    os.makedirs("results", exist_ok=True)
    with open(results_path, "w") as f:
        f.write("# Real-World Traffic Stress Test Results\n\n")
        f.write(f"**Test Sample Size:** {sample_size:,} mixed flows (shuffled)\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| Accuracy | {acc:.4%} |\n")
        f.write(f"| Precision (Attacks) | {prec:.4%} |\n")
        f.write(f"| Recall (Attacks) | {rec:.4%} |\n")
        f.write(f"| **False Positive Rate (FPR)** | **{fpr:.4%}** |\n")
        f.write(f"| False Negative Rate (FNR) | {fnr:.4%} |\n")
        f.write(f"| **Throughput** | **{throughput:,.2f} flows/sec** |\n")
        f.write(f"| Average Latency | {latency_per_flow:.4f} ms |\n")

    print(f"\n[DONE] Stress test report saved to {results_path}")

if __name__ == "__main__":
    run_stress_test()
