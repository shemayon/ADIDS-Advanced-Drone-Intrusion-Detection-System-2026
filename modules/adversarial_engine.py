"""
modules/adversarial_engine.py  —  A-DIDS Adversarial Resilience (Phase 6)
Implements FGSM (Fast Gradient Sign Method) for tabular data using NumPy/pandas.
Works without TensorFlow — uses finite-difference gradient approximation
against the XGBoost model's predict_proba output.

Usage:
    from modules.adversarial_engine import AdversarialEngine
    adv = AdversarialEngine(model)
    perturbed = adv.fgsm(feature_dict, epsilon=0.05)
    evasion_rate = adv.evaluate_robustness(feature_dicts, epsilon=0.05)
"""

from __future__ import annotations
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import FEATURES, FGSM_EPSILON


class AdversarialEngine:
    """
    Adversarial attack & robustness evaluation for XGBoost-based IDS.
    Uses finite-difference gradient estimation (no autograd required).
    """

    def __init__(self, model, epsilon: float = FGSM_EPSILON):
        """
        Parameters
        ----------
        model   : the trained XGBoost / sklearn-compatible model
        epsilon : default FGSM perturbation magnitude
        """
        self.model   = model
        self.epsilon = epsilon
        self.features = FEATURES
        print(f"[Adversarial] Engine ready. Default epsilon={epsilon}")

    def _predict_proba_attack(self, feature_dict: Dict[str, Any]) -> float:
        """Return probability of ATTACK (class 1) for a single flow."""
        row = pd.DataFrame([{f: feature_dict.get(f, 0.0) for f in self.features}])
        return float(self.model.predict_proba(row)[0][1])

    def _gradient(self, feature_dict: Dict[str, Any], delta: float = 1e-4) -> np.ndarray:
        """
        Finite-difference gradient of P(attack) with respect to each feature.
        ∂P/∂x_i ≈ [P(x + δ·e_i) - P(x)] / δ
        """
        base_vals = np.array([feature_dict.get(f, 0.0) for f in self.features])
        base_p    = self._predict_proba_attack(feature_dict)
        grad      = np.zeros_like(base_vals)

        for i in range(len(self.features)):
            perturbed = base_vals.copy()
            perturbed[i] += delta
            p_dict_plus = dict(zip(self.features, perturbed))
            grad[i] = (self._predict_proba_attack(p_dict_plus) - base_p) / delta

        return grad

    def fgsm(
        self,
        feature_dict: Dict[str, Any],
        epsilon: float = None,
        evasion: bool = True,
    ) -> Dict[str, float]:
        """
        Generate an FGSM adversarial example.

        Parameters
        ----------
        feature_dict : original flow features
        epsilon      : perturbation magnitude (defaults to self.epsilon)
        evasion      : if True, perturb to evade (minimize P(attack));
                       if False, amplify (maximize P(attack))

        Returns
        -------
        Perturbed feature dict with same keys as input
        """
        eps  = epsilon if epsilon is not None else self.epsilon
        grad = self._gradient(feature_dict)

        # Evasion: subtract gradient sign to reduce P(attack)
        # Amplification: add gradient sign to increase P(attack)
        direction = -1 if evasion else 1
        sign_grad = np.sign(grad) * direction

        base_vals   = np.array([feature_dict.get(f, 0.0) for f in self.features])
        perturbed   = base_vals + eps * sign_grad

        return {f: float(v) for f, v in zip(self.features, perturbed)}

    def evaluate_robustness(
        self,
        feature_dicts: List[Dict[str, Any]],
        epsilon: float = None,
    ) -> Dict[str, Any]:
        """
        Compute the evasion rate: fraction of attack flows that escape
        detection after adversarial perturbation.

        Returns
        -------
        dict: epsilon, tested, evaded, evasion_rate
        """
        eps     = epsilon if epsilon is not None else self.epsilon
        evaded  = 0
        tested  = 0

        for fd in feature_dicts:
            # Only test flows the model currently classifies as attacks
            orig_p = self._predict_proba_attack(fd)
            if orig_p < 0.5:
                continue
            tested += 1
            perturbed = self.fgsm(fd, epsilon=eps, evasion=True)
            new_p     = self._predict_proba_attack(perturbed)
            if new_p < 0.5:
                evaded += 1

        return {
            "epsilon":      eps,
            "tested":       tested,
            "evaded":       evaded,
            "evasion_rate": evaded / tested if tested else 0.0,
        }


if __name__ == "__main__":
    import joblib
    model  = joblib.load("models/model.pkl")
    engine = AdversarialEngine(model)
    dummy  = {f: 1.0 for f in FEATURES}
    perturbed = engine.fgsm(dummy)
    print("Original:", {f: 1.0 for f in FEATURES})
    print("Perturbed:", perturbed)
