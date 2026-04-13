"""
modules/data_loader.py  —  A-DIDS Data Connector
Production loader that reads from the validated Parquet dataset.
Supports train/test split retrieval and live-feed simulation for
air-gapped or offline testing environments.
"""

from __future__ import annotations
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import FEATURES, DATA_PATH, TEST_SIZE, RANDOM_STATE, LABEL_COL


class A_DIDS_DataLoader:
    """
    Data connector for A-DIDS.
    Primary source: drone_dataset.parquet (2.9M rows, 17 features + label).
    Fallback: statistical simulation for air-gapped / demo environments.
    """

    def __init__(self, dataset_path: str = DATA_PATH):
        self.dataset_path = dataset_path
        self.features     = FEATURES
        self._df: pd.DataFrame = None

    # ── Real Data ─────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Load the full Parquet dataset into memory."""
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at '{self.dataset_path}'. "
                "Run data_pipeline.py first."
            )
        print(f"[DataLoader] Loading: {self.dataset_path}")
        self._df = pd.read_parquet(self.dataset_path)
        print(f"[DataLoader] Shape: {self._df.shape}  |  "
              f"Attack rate: {self._df['label'].mean():.2%}")
        return self._df

    def get_X_y(self, target_col: str = LABEL_COL) -> Tuple[pd.DataFrame, pd.Series]:
        """Return feature matrix X and label series y."""
        if self._df is None:
            self.load()
        X = self._df[self.features]
        y = self._df[target_col]
        return X, y

    def get_train_test_split(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
        target_col: str = LABEL_COL,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Return stratified train/test split."""
        X, y = self.get_X_y(target_col=target_col)
        return train_test_split(X, y,
                                test_size=test_size,
                                random_state=random_state,
                                stratify=y)

    def sample(self, n: int = 100, attack_only: bool = False) -> pd.DataFrame:
        """Return a random sample from the dataset."""
        if self._df is None:
            self.load()
        df = self._df[self._df["label"] == 1] if attack_only else self._df
        return df.sample(min(n, len(df)), random_state=RANDOM_STATE)

    # ── Simulation (air-gapped / offline) ─────────────────────

    def simulate_live_feed(
        self,
        batch_size: int = 1,
        is_attack: bool = False,
    ) -> pd.DataFrame:
        """
        Generate simulated flow data matching the real feature distribution.
        Used for pipeline verification without dataset access.
        """
        rng  = np.random.default_rng(seed=42)
        data = rng.random((batch_size, len(self.features))) * 100

        if is_attack:
            data[:, 0] *= 5.0   # Amplify Duration
            data[:, 2] *= 3.0   # Amplify Entropy
            data[:, 1] *= 10.0  # Amplify Rate

        df = pd.DataFrame(data, columns=self.features)
        df["label"] = int(is_attack)
        return df


if __name__ == "__main__":
    loader = A_DIDS_DataLoader()
    X, y   = loader.get_X_y()
    print(f"Feature shape: {X.shape}")
    print(f"Attack rate  : {y.mean():.2%}")
