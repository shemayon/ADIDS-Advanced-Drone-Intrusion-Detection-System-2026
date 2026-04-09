import pandas as pd
import numpy as np
import os

class A_DIDS_DataLoader:
    def __init__(self, dataset_path=None):
        """
        Data connector for A-DIDS. 
        Bridges the gap between synthetic simulation and real-world 10GB datasets.
        """
        self.dataset_path = dataset_path
        self.feature_count = 62 # standard for our TSLT-Net config
        
    def load_real_data(self, file_name):
        """
        Attempts to load a specific CSV from the provided dataset path.
        """
        if not self.dataset_path:
            print("[WARNING] No dataset path provided. Falling back to Live Sensor Simulation.")
            return self.simulate_live_feed()
            
        full_path = os.path.join(self.dataset_path, file_name)
        if os.path.exists(full_path):
            print(f"[INFO] Loading real telemetry from {full_path}...")
            # Load only a subset for efficiency in detection tasks
            df = pd.read_csv(full_path, nrows=100)
            # Preprocessing would go here (normalization/scaling)
            return df.values[:, :self.feature_count]
        else:
            print(f"[ERROR] Data file {file_name} not found at {self.dataset_path}.")
            return self.simulate_live_feed()

    def simulate_live_feed(self, batch_size=1, is_attack=False):
        """
        Generates realistic synthetic data matching the TSLT-Net input shape.
        Used for air-gapped testing and pipeline verification.
        """
        data = np.random.randn(batch_size, self.feature_count).astype(np.float32)
        if is_attack:
            # Inject anomalies into the synthetic flow to trigger the IDS
            data[:, 0] = data[:, 0] * 5.0 # Payload Anomaly
            data[:, 1] = data[:, 1] * 7.0 # Frequency Anomaly
        return data

if __name__ == "__main__":
    loader = A_DIDS_DataLoader()
    sample = loader.simulate_live_feed(is_attack=True)
    print(f"A-DIDS Data Connector active. Simulated Feed Shape: {sample.shape}")
