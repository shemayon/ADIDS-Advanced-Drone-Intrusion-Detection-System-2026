"""
modules/apex_engine.py  —  The Ultimate Unified Defense Orchestrator (Phase 8)
Brings together RF-PHY, Zero-Day, Temporal Memory, Swarm Consensus, 
and Kinetic Response under one roof alongside the XGBoost core.
"""
from typing import Dict, Any, List

from modules.zero_day_detector import ZeroDayDetector
from modules.kinetic_response import KineticResponseFramework
from modules.swarm_consensus import SwarmConsensus
from modules.rf_phy_layer import RFPhysicalLayer
from modules.temporal_memory import TemporalMemory

class ApexEngine:
    def __init__(self):
        print("\n\n[APEX] 🛡️ INITIALIZING HYBRID SENTINEL ARCHITECTURE 🛡️")
        self.zero_day = ZeroDayDetector()
        self.kinetic = KineticResponseFramework()
        self.consensus = SwarmConsensus(total_nodes=5, required_consensus=3)
        self.rf_phy = RFPhysicalLayer()
        self.temporal = TemporalMemory(window_size=10)
        self.system_ready = True
        
    def train_unsupervised(self, data):
        """Prepare Zero-Day detector."""
        # Only fit on normal traffic for anomaly isolation
        normal_data = data[data["label"] == 0]
        self.zero_day.train_on_benign(normal_data)
        
    def orchestrate(
        self, 
        flow: Dict[str, float], 
        xgboost_is_attack: bool, 
        xgboost_confidence: float, 
        alert_type: str
    ) -> Dict[str, Any]:
        """
        Takes a single network flow and the base XGBoost prediction,
        running it through the ultimate validation chain.
        """
        print("\n  [APEX] Cross-Referencing Defense Matrix...")
        
        # 1. Temporal Check
        escalation = self.temporal.add_flow(flow)
        
        # 2. Zero-Day Check (what if XGBoost missed it?)
        is_zero_day = False
        if not xgboost_is_attack:
            is_zero_day = self.zero_day.detect(flow)
            if is_zero_day:
                print(f"  [ZeroDay] ⚠️ UNKNOWN ANOMALY DETECTED BYPASSING XGBOOST.")
                alert_type = "Zero-Day Exploit"
                xgboost_confidence = 0.99
                xgboost_is_attack = True
                
        # 3. RF-PHY Check
        phy_status = self.rf_phy.analyze_spectrum(noise_floor_dbm=-30.0 if alert_type == "Jamming" else -90.0)
        
        # 4. Swarm Consensus Check
        if xgboost_is_attack:
            verified = self.consensus.verify_alert(alert_type, xgboost_confidence)
            
            # 5. Automated Kinetic Execution
            if verified or phy_status != "CLEAN" or escalation > 0.8:
                self.kinetic.execute_mitigation(alert_type)
                return {"action_taken": "KINETIC_MITIGATION", "reason": "Consensus Verified"}
            else:
                return {"action_taken": "LOG_ONLY", "reason": "No Consensus or Escalation"}
                
        return {"action_taken": "NONE", "reason": "Traffic Benign"}
