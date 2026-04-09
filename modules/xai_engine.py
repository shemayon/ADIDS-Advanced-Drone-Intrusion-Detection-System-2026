import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ADIDS_XAI:
    def __init__(self, model, feature_names=None):
        """
        XAI Module for Advanced Drone IDS.
        
        Args:
            model: The TSLT-Net model instance.
            feature_names (list): Names of the network features for better visualization.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None

    def initialize_explainer(self, background_data):
        """
        Initializes the SHAP Explainer.
        Uses KernelExplainer for broad compatibility across different TF versions.
        """
        # KernelExplainer is more stable with the latest Keras/TF Operations
        # We wrap model.predict to ensure standard input format
        def predict_wrapper(data):
            return self.model.predict(data, verbose=0)
            
        self.explainer = shap.KernelExplainer(predict_wrapper, background_data)

    def explain_sample(self, sample):
        """
        Calculates SHAP values for a single sample or batch.
        
        Args:
            sample (np.ndarray): The input flow sample to explain.
            
        Returns:
            np.ndarray: SHAP values.
        """
        if self.explainer is None:
            raise ValueError("XAI Explainer not initialized. Call initialize_explainer() first.")
        
        shap_values = self.explainer.shap_values(sample)
        return shap_values

    def plot_summary(self, samples, save_path=None):
        """
        Generates a SHAP summary plot showing feature importance.
        """
        shap_values = self.explain_sample(samples)
        
        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, samples, feature_names=self.feature_names, show=False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"XAI Summary plot saved to {save_path}")
        else:
            plt.show()

    def get_top_features_for_sample(self, sample, top_n=3):
        """
        Returns the top N features that contributed to the model's decision for a specific flow.
        """
        shap_values = self.explain_sample(sample.reshape(1, -1))
        
        # Flatten shap_values for binary classification output (usually [1, features, 1])
        if isinstance(shap_values, list):
            sv = shap_values[0].flatten()
        else:
            sv = shap_values.flatten()
            
        indices = np.argsort(np.abs(sv))[-top_n:][::-1]
        
        contributions = []
        for i in indices:
            feat_name = self.feature_names[i] if self.feature_names else f"Feature_{i}"
            contributions.append({"feature": feat_name, "shap_value": sv[i]})
            
        return contributions

if __name__ == "__main__":
    print("A-DIDS XAI Engine initialized. Ready for model interpretation.")
