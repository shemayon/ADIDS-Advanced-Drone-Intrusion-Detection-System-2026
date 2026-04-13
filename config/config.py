"""
config/config.py  —  A-DIDS Central Configuration
All paths, feature names, and deployment constants live here.
Import this module from any other script to stay in sync.
"""

import os

# ── Feature Set (must match drone_dataset.parquet columns) ───
FEATURES = [
    "Duration", "Rate", "Entropy",
    "Payload_Length", "Var_Payload",
    "syn_flag_number", "ack_flag_number",
    "rst_flag_number", "fin_flag_number",
    "TCP", "UDP",
    "Number", "Tot size",
    "AVG", "Std", "Min", "Max",
]

# ── Label ──────────────────────────────────────────────────────
LABEL_COL = "label"
CLASS_LABEL_COL = "class_label"
BENIGN_DIRS = {"Regular", "Video"}          # dirs treated as benign

# Multi-class mapping based on discovered folder structure
ATTACK_CLASSES = {
    "Regular": 0,
    "Video": 0,
    "DoS": 1,
    "Injection": 2,
    "Manipulation": 3,
    "MITM": 4,
    "Password Cracking": 5,
    "Replay": 6,
    "Ip Spoofing": 7,
    "Unauth": 8
}
ATTACK_LABEL = 1
BENIGN_LABEL = 0

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "drone_dataset.parquet")

# ── Model Hyperparameters (used at training time) ─────────────
XGB_PARAMS = {
    "n_estimators":  200,
    "max_depth":     6,
    "learning_rate": 0.1,
    "n_jobs":        -1,
    "eval_metric":   "logloss",
    "verbosity":     1,
}
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ── PCAP Inference Defaults ───────────────────────────────────
PCAP_MAX_PACKETS = 50_000   # packets to read per file (0 = all)
PCAP_MIN_FLOW_PKTS = 3      # discard flows smaller than this

# ── XAI ───────────────────────────────────────────────────────
SHAP_BACKGROUND_SAMPLES = 200   # rows sampled from training set for SHAP

# ── Adversarial ───────────────────────────────────────────────
FGSM_EPSILON = 0.05             # perturbation step size

# ── Tactical Deployment ───────────────────────────────────────
DEPLOYMENT_UNIT = "UAE-Tactical-Swarm-01"
REGION          = "Abu Dhabi Defence Sector"
