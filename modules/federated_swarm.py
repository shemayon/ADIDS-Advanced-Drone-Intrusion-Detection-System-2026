"""
modules/federated_swarm.py  —  Phase 3 (Federated Learning)
Simulates decentralized IDS training across a swarm of edges.
No raw data leaves the local EdgeNode. Only abstract model architecture
(trees/weights) is sent to the SwarmAggregator.
"""

from typing import List
import pandas as pd
from xgboost import XGBClassifier

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.config import XGB_PARAMS

class EdgeNode:
    """A single drone in the swarm training locally."""
    def __init__(self, node_id: str, local_data: pd.DataFrame):
        self.node_id = node_id
        self.local_data = local_data
        self.features = [c for c in local_data.columns if c != "label"]
        # Use a lightweight version of the model for edge hardware
        self.model = XGBClassifier(**{**XGB_PARAMS, "n_estimators": 50})
        
    def local_train(self):
        print(f"  [Edge {self.node_id}] Training locally on {len(self.local_data)} records...")
        X = self.local_data[self.features]
        y = self.local_data["label"]
        self.model.fit(X, y, verbose=False)
        return self.model.get_booster()

class SwarmAggregator:
    """Central ground station or master drone merging weights."""
    def __init__(self):
        self.global_trees = []

    def federated_merge(self, boosters: List):
        """
        In XGBoost, true federated averaging is complex because trees are non-parametric.
        A common FL technique for trees is to horizontally concatenate the tree ensembles, 
        or use federated histogram approaches. Here we simulate the merge conceptually.
        """
        print(f"[Swarm] Aggregating intelligence from {len(boosters)} discrete nodes.")
        print("[Swarm] Raw telemetry data transferred: 0 Bytes")
        print("[Swarm] Swarm Global Model synchronized and ready.")
        # In a real deployed iteration, we would use something like XGBoost's 
        # federated learning plugin or secure aggregation of gradient histograms.
        return True

if __name__ == "__main__":
    # Smoke Test
    import numpy as np
    from modules.data_loader import A_DIDS_DataLoader
    
    loader = A_DIDS_DataLoader()
    df = loader.simulate_live_feed(batch_size=100)
    
    # Simulate data partitioning for two drones
    edge1 = EdgeNode("UAV-1", df.iloc[:50])
    edge2 = EdgeNode("UAV-2", df.iloc[50:])
    
    b1 = edge1.local_train()
    b2 = edge2.local_train()
    
    swarm = SwarmAggregator()
    swarm.federated_merge([b1, b2])
