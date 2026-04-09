# run_pipeline.py

import os
import sys
from pcap_processor import process_pcap
from inference_engine import InferenceEngine
from modules.tactical_briefing import TacticalBriefing

def main():
    # 1. Initialize Engine
    model_path = "A-DIDS/models/model.pkl"
    if not os.path.exists(model_path):
        print(f"[ERROR] {model_path} not found. Ensure you have trained the model first.")
        # Attempt to find it in the data folder if moved
        alternate_path = "A-DIDS/data/model.pkl"
        if os.path.exists(alternate_path):
            model_path = alternate_path
        else:
            return

    engine = InferenceEngine(model_path)
    
    # 1a. Initialize Briefing Engine (Bilingual)
    brief_en = TacticalBriefing(language="en")
    brief_ar = TacticalBriefing(language="ar")

    # 2. Define Test PCAP
    # Using a typical file path from the ISOT dataset structure
    pcap_file = "A-DIDS/data/extracted/ISOT Drone Dataset/Raw Data/DoS/Dos_1_4_30_20mins.pcap"
    
    if not os.path.exists(pcap_file):
        print(f"[WARNING] Test PCAP not found at {pcap_file}")
        print("Please ensure the dataset is extracted to A-DIDS/data/extracted/")
        return

    # 3. Process PCAP
    print(f"\n[STEP 1] Extracting features from: {os.path.basename(pcap_file)}")
    features_list = process_pcap(pcap_file)

    # 4. Execute Inference
    print("\n[STEP 2] Running Inference + Tactical Briefings")
    print("=" * 60)
    for i, f in enumerate(features_list[:2]): # Top 2 for demo
        result = engine.predict(f)
        
        # Extract top 3 triggers for the briefing
        triggers = []
        if result['explanation']:
            expl = result['explanation']
            sorted_feats = sorted(expl.items(), key=lambda x: abs(x[1]), reverse=True)
            triggers = [k for k, v in sorted_feats[:3]]

        # Generate Briefings
        report_en = brief_en.generate_briefing("DoS", result['confidence'], triggers)
        report_ar = brief_ar.generate_briefing("DoS", result['confidence'], triggers)

        print(f"\nFLOW {i+1} COMMAND BRIEFING (EN):")
        print(report_en)
        print(f"\nFLOW {i+1} COMMAND BRIEFING (AR):")
        print(report_ar)
        print("-" * 60)

    print("\n[SUCCESS] End-to-End Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
