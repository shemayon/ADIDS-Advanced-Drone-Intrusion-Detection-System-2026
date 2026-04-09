import os
import sys
import numpy as np

# Ensure root (A-DIDS) is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modules.ids_engine import build_ids_model
from modules.adversarial_engine import AdversarialEngine
from config.config import ISOT_FEATURE_NAMES

def run_phase2_verification():
    print("=== A-DIDS Phase 2: Adversarial Resilience Verification ===\n")
    
    # 1. Setup Model
    input_dim = len(ISOT_FEATURE_NAMES)
    model = build_ids_model(input_dim)
    adv_engine = AdversarialEngine(model)
    
    # 2. Simulate a baseline "Attack" sample that the model detects
    # In a real scenario, this would be a DoS flow.
    # For simulation, we create a sample and "detect" it.
    sample = np.random.randn(1, input_dim).astype(np.float32)
    sample[0][1] = 10.0 # Extreme anomaly
    
    pred_orig = model.predict(sample, verbose=0)[0][0]
    print(f"[STAGE 1] Original Prediction Score: {pred_orig:.4f}")
    print(f"Status: {'CRITICAL ALERT' if pred_orig > 0.5 else 'NORMAL (False Negative)'}")
    
    # 3. Generate Adversarial Evasion Sample (The "Stealth" Attack)
    print("\n[STAGE 2] Generating Adversarial Evasion (FGSM)...")
    epsilon = 0.2 # Perturbation strength
    adv_sample = adv_engine.generate_fgsm_sample(sample, epsilon=epsilon)
    
    pred_adv = model.predict(adv_sample, verbose=0)[0][0]
    print(f"Adversarial Prediction Score: {pred_adv:.4f}")
    
    delta = pred_orig - pred_adv
    if delta > 0:
        print(f"SUCCESS: Adversarial Evasion reduced detection score by {delta:.4f}")
        if pred_adv < 0.5 < pred_orig:
            print(">>> CRITICAL: Attack is now STEALTHY (Model fooled!)")
    else:
        print("INFO: Model was robust to this specific perturbation.")

    # 4. Demonstrate Defensive Hardening (Adversarial Training)
    print("\n[STAGE 3] Demonstration: Defensive Hardening Step...")
    import tensorflow as tf
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # labels: 1 for attack
    labels = np.array([[1.0]], dtype=np.float32)
    
    loss = adv_engine.adversarial_training_step(sample, labels, optimizer, epsilon=epsilon)
    print(f"Hardening step complete. Training Loss: {loss:.4f}")
    print("The model is now learning to recognize the 'Stealth' variant of the attack.\n")
    
    print("=== PHASE 2 VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    try:
        run_phase2_verification()
    except Exception as e:
        print(f"\n[ERROR] Phase 2 verification failed: {e}")
        sys.exit(1)
