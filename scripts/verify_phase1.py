import os
import sys
import numpy as np

# Ensure root (A-DIDS) is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modules.ids_engine import build_ids_model
from modules.xai_engine import ADIDS_XAI
from modules.tactical_briefing import TacticalBriefing
from config.config import ISOT_FEATURE_NAMES

def run_phase1_verification():
    print("=== A-DIDS Phase 1 System Verification ===\n")
    
    # 1. Test IDS Engine
    input_dim = len(ISOT_FEATURE_NAMES)
    print(f"[TEST 1/3] Building TSLT-Net Engine (Input Dim: {input_dim})...")
    model = build_ids_model(input_dim)
    print("SUCCESS: Engine built.\n")
    
    # 2. Test XAI Engine
    print("[TEST 2/3] Initializing XAI Engine and Explaining Sample...")
    # Simulate a background and an attack sample
    # input_dim is derived from config (should be 62)
    background = np.random.randn(5, input_dim) 
    attack_sample = np.random.randn(input_dim)
    attack_sample[1] = 5.0 # Simulate anomaly in feature 1 (Packet Rate)
    
    xai = ADIDS_XAI(model, feature_names=ISOT_FEATURE_NAMES)
    xai.initialize_explainer(background)
    top_features_data = xai.get_top_features_for_sample(attack_sample)
    
    top_feat_names = [f['feature'] for f in top_features_data]
    print(f"SUCCESS: XAI triggered. Top features detected: {top_feat_names}\n")
    
    # 3. Test Tactical Briefing
    print("[TEST 3/3] Generating Narrative Briefing (Multilingual)...")
    briefing_gen = TacticalBriefing(language="en")
    report_en = briefing_gen.generate_briefing("DoS", 0.99, top_feat_names)
    
    briefing_gen_ar = TacticalBriefing(language="ar")
    # Translate features manually for a clean AR demo in CLI
    top_feat_ar = ["معدل الحزم", "حجم البيانات"] 
    report_ar = briefing_gen_ar.generate_briefing("DoS", 0.99, top_feat_ar)
    
    print("SUCCESS: English Report Generated:\n")
    print(report_en)
    print("\nSUCCESS: Arabic Report Generated:\n")
    print(report_ar)
    
    print("\n=== SYSTEM VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    try:
        run_phase1_verification()
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        sys.exit(1)
