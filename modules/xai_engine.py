"""
modules/xai_engine.py  —  A-DIDS Explainability Engine (SHAP)
Uses SHAP TreeExplainer (native XGBoost support, no kernel approximation)
to generate Truth Triggers for every alert.

Usage:
    from modules.xai_engine import XAI_Engine
    xai = XAI_Engine(model, X_background)
    triggers = xai.get_top_features(feature_dict, top_n=3)
"""

from __future__ import annotations
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import FEATURES

# Import SHAP — installed as part of project requirements
import shap


class XAI_Engine:
    """
    SHAP-based explanation engine for the XGBoost IDS model.
    TreeExplainer is exact (no sampling) and very fast for XGBoost.
    """

    def __init__(self, model, background_data: pd.DataFrame = None):
        """
        Parameters
        ----------
        model           : the trained XGBoost model (joblib-loaded)
        background_data : optional DataFrame for feature baseline;
                          if None, uses the model's tree structure directly
        """
        self.model    = model
        self.features = FEATURES

        # TreeExplainer is the correct explainer for XGBoost — exact values
        if background_data is not None:
            self.explainer = shap.TreeExplainer(model,
                                                data=background_data[FEATURES])
        else:
            self.explainer = shap.TreeExplainer(model)

        print("[XAI] SHAP TreeExplainer ready.")

    def explain(self, feature_dict: Dict[str, Any]) -> Dict[str, float]:
        """
        Returns a dict mapping feature_name → shap_value for a single flow.
        Positive values push toward ATTACK, negative toward BENIGN.
        """
        row   = pd.DataFrame([{f: feature_dict.get(f, 0.0) for f in FEATURES}])
        shap_values = self.explainer.shap_values(row)

        # For binary XGBoost, shap_values is a 2D array [1 x n_features]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]          # index 1 = positive class
        else:
            sv = shap_values[0]

        return dict(zip(self.features, sv.tolist()))

    def get_top_features(
        self,
        feature_dict: Dict[str, Any],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Returns the top_n features that most influenced the prediction.

        Returns
        -------
        list of { feature, shap_value, direction }
        """
        shap_map = self.explain(feature_dict)
        sorted_feats = sorted(shap_map.items(),
                              key=lambda x: abs(x[1]), reverse=True)
        return [
            {
                "feature":     feat,
                "shap_value":  round(val, 6),
                "direction":   "→ ATTACK" if val > 0 else "→ BENIGN",
            }
            for feat, val in sorted_feats[:top_n]
        ]

    def explain_batch(self, feature_dicts: List[Dict[str, Any]]) -> np.ndarray:
        """Batch SHAP values — returns array of shape [N, n_features]."""
        rows = pd.DataFrame(
            [{f: d.get(f, 0.0) for f in FEATURES} for d in feature_dicts]
        )
        sv = self.explainer.shap_values(rows)
        return sv[1] if isinstance(sv, list) else sv


if __name__ == "__main__":
    import joblib
    model = joblib.load("models/model.pkl")
    xai   = XAI_Engine(model)
    dummy = {f: 0.0 for f in FEATURES}
    print(xai.get_top_features(dummy))
