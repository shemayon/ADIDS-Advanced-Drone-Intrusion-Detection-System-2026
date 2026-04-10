"""
modules/zero_day_detector.py  —  Phase 8: Unsupervised Anomaly Detection
Uses Isolation Forest to detect novel zero-day attacks that bypass supervised learning.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import FEATURES

class ZeroDayDetector:
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(
            n_estimators=100, 
            contamination=contamination, 
            random_state=42, 
            n_jobs=-1
        )
        self.is_trained = False
        
    def train_on_benign(self, data: pd.DataFrame):
        """Train ONLY on benign traffic so anything else is an anomaly."""
        print("[ZeroDay] Training Isolation Forest on Normal Flight Patterns...")
        self.model.fit(data[FEATURES])
        self.is_trained = True
        
    def detect(self, feature_dict: dict) -> bool:
        """Returns True if anomaly (Zero-Day) detected."""
        if not self.is_trained:
            return False
            
        row = pd.DataFrame([{f: feature_dict.get(f, 0.0) for f in FEATURES}])
        # predict returns 1 for inliers, -1 for outliers
        return self.model.predict(row)[0] == -1

if __name__ == "__main__":
    zd = ZeroDayDetector()
    print("Zero-Day Detector initialized.")
