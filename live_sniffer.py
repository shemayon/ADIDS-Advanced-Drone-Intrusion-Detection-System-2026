"""
live_sniffer.py  —  Real-Time Drone IDS Streaming Pipeline
Captures live network packets, reconstructs flows, and performs
millisecond-latency inference with the A-DIDS XGBoost classifier.
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import joblib
from scapy.all import sniff, IP, TCP, UDP
from collections import defaultdict

# Add current directory to path for modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import FEATURES, MODEL_PATH, ATTACK_CLASSES

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("A-DIDS-Live")

class LiveIDS:
    def __init__(self, model_path, flow_timeout=30):
        self.flow_timeout = flow_timeout
        self.flows = {} # key: (src, dst, sport, dport, proto) -> flow_data
        
        # Load Model
        logger.info(f"Loading model: {model_path}")
        self.model = joblib.load(model_path)
        
        # Performance Tracking
        self.pkt_count = 0
        self.flow_count = 0
        self.start_time = time.time()
        self.prediction_log = []

    def get_flow_key(self, pkt):
        """Generates a 5-tuple key for flow tracking."""
        if IP in pkt:
            src = pkt[IP].src
            dst = pkt[IP].dst
            proto = pkt[IP].proto
            sport = 0
            dport = 0
            if TCP in pkt:
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
            elif UDP in pkt:
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport
            
            # Form bi-directional key (sort IPs to ensure consistency)
            ips = sorted([src, dst])
            ports = sorted([sport, dport])
            return (ips[0], ips[1], ports[0], ports[1], proto)
        return None

    def process_packet(self, pkt):
        self.pkt_count += 1
        key = self.get_flow_key(pkt)
        if not key:
            return

        now = time.time()
        if key not in self.flows:
            # Initialize New Flow
            self.flows[key] = {
                "start": now,
                "last": now,
                "pkt_count": 1,
                "byte_count": len(pkt),
                "src": pkt[IP].src,
                "dst": pkt[IP].dst,
                "proto": pkt[IP].proto,
                "iat": []
            }
        else:
            # Update Existing Flow
            f = self.flows[key]
            f["iat"].append((now - f["last"]) * 1000) # ms
            f["last"] = now
            f["pkt_count"] += 1
            f["byte_count"] += len(pkt)

        # Check for timeouts (simulation of a finalized flow)
        if self.pkt_count % 100 == 0:
            self.cleanup_and_predict(now)

    def cleanup_and_predict(self, current_time):
        to_delete = []
        for key, f in self.flows.items():
            duration = f["last"] - f["start"]
            if (current_time - f["last"] > self.flow_timeout) or (f["pkt_count"] > 50):
                self.predict_flow(key, f)
                to_delete.append(key)
        
        for key in to_delete:
            del self.flows[key]

    def predict_flow(self, key, f):
        self.flow_count += 1
        duration = f["last"] - f["start"]
        avg_iat = np.mean(f["iat"]) if f["iat"] else 0
        
        # Mapping live captures to the specific feature set used in training
        # Note: This is a subset/mapping of the 63 features down to the 19 calibrated ones
        features = {
            "Flow Duration": duration,
            "Total Fwd Packets": f["pkt_count"] // 2 + 1,
            "Total Backward Packets": f["pkt_count"] // 2,
            "Flow Bytes/s": f["byte_count"] / duration if duration > 0 else 0,
            "Flow Packets/s": f["pkt_count"] / duration if duration > 0 else 0,
            "Flow IAT Mean": avg_iat,
            "Fwd IAT Mean": avg_iat, 
            "Bwd IAT Mean": avg_iat,
            "Fwd Header Length": 20 * (f["pkt_count"] // 2 + 1),
            "Bwd Header Length": 20 * (f["pkt_count"] // 2),
            "Fwd Packets/s": (f["pkt_count"] // 2 + 1) / duration if duration > 0 else 0,
            "Bwd Packets/s": (f["pkt_count"] // 2) / duration if duration > 0 else 0,
            "Min Packet Length": 0,
            "Max Packet Length": 1500,
            "Packet Length Mean": f["byte_count"] / f["pkt_count"],
            "Average Packet Size": f["byte_count"] / f["pkt_count"],
            "Subflow Fwd Bytes": f["byte_count"] // 2,
            "Init_Win_bytes_forward": 8192,
            "Init_Win_bytes_backward": 8192
        }

        # Convert to DataFrame
        df = pd.DataFrame([features])[FEATURES]
        
        # Perform Inference
        infer_start = time.time()
        pred = self.model.predict(df)[0]
        prob = self.model.predict_proba(df)[0].max()
        infer_time = (time.time() - infer_start) * 1000 # ms
        
        status = "!!! ATTACK DETECTED !!!" if pred == 1 else "BENIGN"
        logger.info(f"[{status}] Flow: {key[0]} <-> {key[1]} | Confidence: {prob:.2%} | Latency: {infer_time:.4f}ms")
        
        self.prediction_log.append({
            "timestamp": time.ctime(),
            "src": f["src"],
            "dst": f["dst"],
            "result": status,
            "confidence": prob,
            "latency_ms": infer_time
        })

    def run(self, interface=None):
        logger.info(f"Starting Live IDS Sniffer on {interface if interface else 'default interface'}...")
        try:
            sniff(iface=interface, prn=self.process_packet, store=0)
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        duration = time.time() - self.start_time
        logger.info("\n" + "="*40)
        logger.info("  LIVE SESSION SUMMARY")
        logger.info("="*40)
        logger.info(f"Total Packets: {self.pkt_count}")
        logger.info(f"Total Flows  : {self.flow_count}")
        logger.info(f"Avg Pkt/sec  : {self.pkt_count / duration:.2f}")
        logger.info(f"Avg Inf/sec  : {self.flow_count / duration:.2f}")
        
        # Save logs
        log_df = pd.DataFrame(self.prediction_log)
        log_df.to_csv("results/live_logs.csv", index=False)
        logger.info("Session logs saved to results/live_logs.csv")

if __name__ == "__main__":
    ids = LiveIDS(MODEL_PATH)
    # We use None for default interface, or "lo" for testing
    ids.run(interface=None)
