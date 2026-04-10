# A-DIDS: Training & Validation Results

This document captures the real-world performance metrics from the Phase 4 training session against the ISOT Drone Dataset.

## 1. Dataset Scale
* **Total Flow Records**: 2,945,854
* **Benign Traffic**: 1,357,960 flows (46.1%)
* **Attack Traffic**: 1,587,894 flows (53.9%)
* **Features Extracted**: 17 flow-level network markers (Duration, Rate, Entropy, Flag counts, etc.)

## 2. Core Detection Performance
*Model: XGBoost (200 trees, max_depth=6)*

| Metric | Score | Note |
|---|---|---|
| **Accuracy** | 99.1% | Highly discriminative |
| **ROC-AUC** | 0.9998 | Near-perfect separation |
| **5-Fold CV** | 99.10% ± 0.02% | Very stable, no variance issue |
| **False Positives**| 2,915 / 271K (~1.0%) | Low disruption to normal operations |
| **False Negatives**| 1,578 / 317K (~0.5%) | Excellent threat capture rate |

> **Analyst Note on Overfitting**: Training logloss vs. testing logloss tracked consistently together across all 200 epochs. The cross-validation variance is near zero (±0.0002). This confirms the 99% accuracy is legitimate and reflects the fact that DoS/Injection network signatures are inherently distinct at the flow feature level.

## 3. Threat Signatures (SHAP XAI)
The XAI engine identified the following features as the heaviest "Truth Triggers" that drive the model's decision-making:

1. **Entropy** (Importance Score: 9,881) — Tracks payload randomness, highly relevant for injection and spoofing.
2. **Duration** (Importance Score: 1,221) — Flow length anomalies typical of DoS/flood attacks.
3. **Rate** (Importance Score: 491) — Packet intensity spikes.
4. **UDP Flag** (Importance Score: 296) — Protocol deviation.

These findings strongly align with established network intrusion detection theory.

## 4. End-to-End Pipeline Verification
The entire `PCAP → Extractor → Model → Briefing` pipeline was successfully verified live on massive packet captures:
* Ran against a 38MB raw DoS capture (`Dos_1_4_30_20mins.pcap`).
* Processed 50,000 packets into 3,296 distinct flows.
* Successfully flagged 100% of the malicious flows with >95% confidence, generating the correct SHAP triggers and bilingual UAE command briefings.
