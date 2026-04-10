"""
modules/ids_engine.py  —  A-DIDS Core IDS Wrapper
Provides a unified interface over the trained XGBoost classifier.
The TSLT-Net architecture described in the project vision is captured
here as the XGBoost baseline (Phase 4 validated, 99% accuracy).
"""

from __future__ import annotations
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from inference_engine import InferenceEngine
from config.config import FEATURES, MODEL_PATH


class IDS_Engine:
    """
    Central IDS dispatch engine.
    Wraps InferenceEngine and exposes:
      - scan_flow(feature_dict)   → single flow result
      - scan_batch(feature_list)  → list of results
      - summary(results)          → aggregate threat summary
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self._engine = InferenceEngine(model_path=model_path)
        print("[IDS] Engine ready.")

    def scan_flow(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Analyse a single flow feature dict."""
        return self._engine.predict(feature_dict)

    def scan_batch(self, feature_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyse a list of flow feature dicts in one batch pass."""
        return self._engine.predict_batch(feature_list)

    def summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate statistics over a list of flow results.

        Returns
        -------
        dict:
          total, attacks, benign, attack_rate,
          high_confidence_attacks (conf >= 0.95)
        """
        total   = len(results)
        attacks = sum(1 for r in results if r["prediction"] == 1)
        benign  = total - attacks
        high    = sum(1 for r in results
                      if r["prediction"] == 1 and r["confidence"] >= 0.95)
        return {
            "total":                    total,
            "attacks":                  attacks,
            "benign":                   benign,
            "attack_rate":              attacks / total if total else 0.0,
            "high_confidence_attacks":  high,
        }

    @property
    def feature_names(self) -> List[str]:
        return FEATURES


if __name__ == "__main__":
    engine = IDS_Engine()
    test   = {f: 0.0 for f in FEATURES}
    result = engine.scan_flow(test)
    print(f"IDS Engine smoke test: {result}")
