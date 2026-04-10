"""
inference_engine.py  —  A-DIDS Inference Engine
Loads the trained XGBoost model and provides single/batch prediction.

Usage:
    from inference_engine import InferenceEngine
    engine = InferenceEngine()
    result = engine.predict(feature_dict)
    results = engine.predict_batch(list_of_feature_dicts)
"""

from __future__ import annotations
import os
import sys
from typing import Any, Dict, List

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import FEATURES, MODEL_PATH


class InferenceEngine:
    """Wraps the XGBoost model for real-time and batch inference."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Run train_model.py first or copy model.pkl to models/."
            )
        self.model = joblib.load(model_path)
        print(f"[Engine] Loaded model: {model_path}")

    def predict(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict on a single flow.

        Returns
        -------
        dict  prediction (0/1), confidence (0-1), label (str)
        """
        row = {f: feature_dict.get(f, 0.0) for f in FEATURES}
        df  = pd.DataFrame([row])[FEATURES]

        pred       = int(self.model.predict(df)[0])
        proba      = self.model.predict_proba(df)[0]
        confidence = float(proba[pred])

        return {
            "prediction": pred,
            "confidence": confidence,
            "label":      "ATTACK" if pred == 1 else "BENIGN",
        }

    def predict_batch(self, feature_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch inference — significantly faster than looping predict()."""
        rows  = [{f: d.get(f, 0.0) for f in FEATURES} for d in feature_dicts]
        df    = pd.DataFrame(rows)[FEATURES]
        preds = self.model.predict(df).tolist()
        probas = self.model.predict_proba(df).tolist()

        return [
            {
                "prediction": int(p),
                "confidence": float(prob[int(p)]),
                "label":      "ATTACK" if p == 1 else "BENIGN",
            }
            for p, prob in zip(preds, probas)
        ]

    @property
    def feature_names(self) -> List[str]:
        return FEATURES


if __name__ == "__main__":
    engine = InferenceEngine()
    # Smoke test with a zero-vector
    result = engine.predict({f: 0.0 for f in FEATURES})
    print(f"Smoke test result: {result}")
