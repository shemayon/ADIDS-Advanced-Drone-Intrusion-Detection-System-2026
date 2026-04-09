# inference_engine.py

import joblib
import pandas as pd
import shap
import os

class InferenceEngine:
    def __init__(self, model_path):
        """
        Inference engine for A-DIDS.
        Loads a pre-trained XGBoost model and provides SHAP explanations.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please run train_model.py first.")
            
        print(f"[INFO] Loading model: {model_path}")
        self.model = joblib.load(model_path)

        try:
            # SHAP Explainer for Tree-based models (XGBoost)
            self.explainer = shap.Explainer(self.model)
            print("[INFO] SHAP Explainer initialized.")
        except Exception as e:
            print(f"[WARNING] SHAP Explainer failed to initialize: {e}")
            self.explainer = None

    def predict(self, features):
        """
        Predicts intrusion status and provides confidence + explanation.
        """
        df = pd.DataFrame([features])

        # Prediction
        pred = int(self.model.predict(df)[0])

        # Confidence Score
        confidence = None
        if hasattr(self.model, "predict_proba"):
            confidence = float(max(self.model.predict_proba(df)[0]))

        # SHAP Values (Explanation)
        explanation = None
        if self.explainer:
            try:
                shap_values = self.explainer(df)
                explanation = dict(zip(df.columns, shap_values.values[0]))
            except Exception as e:
                print(f"[WARNING] SHAP calculation failed: {e}")

        return {
            "prediction": pred,
            "prediction_text": "ATTACK" if pred == 1 else "BENIGN",
            "confidence": confidence,
            "explanation": explanation
        }

if __name__ == "__main__":
    # Test script will be handled by run_pipeline.py
    pass
