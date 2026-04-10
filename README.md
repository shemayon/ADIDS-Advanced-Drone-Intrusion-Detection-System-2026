# A-DIDS: Advanced Drone Intrusion Detection System
*Developed for the 2026 UAE Defence Showcase.*

## 🏛️ Current System Overview (Phase 4 Verified - Production-Ready)
The A-DIDS infrastructure is a **validated defence baseline**, transitioning from a research experiment to a production-grade Intrusion Detection System.

* **Production Telemetry**: Scaled and processed **2.94 Million rows** of real-world ISOT Drone Dataset telemetry.
* **AI Core**: High-precision XGBoost baseline detecting network anomalies (DoS, Spoofing, Injection) with **99.1% accuracy**.
* **Explainability (XAI)**: Integrated SHAP **TreeExplainer** engine providing exact "Truth Triggers" for every tactical alert.
* **Tactical Reporting**: Bilingual (English/Arabic) command briefings generated in real-time, optimized for tactical field operations.
* **Adversarial Hardening**: Built-in FGSM engine (Phase 6 preview) to evaluate and ensure resilience against evasion attacks.
* **High Performance**: Uses Parquet storage, streaming Scapy PCAP processing, and lightweight modular Python for edge-deployment compatibility.

---

## 🚀 Future Enhancements (The Strategic Roadmap)
1. **Phase 5: Falcon-H1 Integration (UAE TII)**
   Moving from report "templates" to a **Live AI Brain** that can reason about complex threat scenarios in native Arabic/English using UAE's Falcon models.
2. **Phase 6: Adversarial Resilience & Hardening**
   Implementing full persistence against AI-targeted attacks (FGSM/PGD) ensuring the IDS cannot be "blinded" by stealthy signals.
3. **Phase 3: Federated Swarm Intelligence**
   Enabling a **Swarm of Drones** to learn from each other's detections decentrally, without uploading raw, sensitive telemetry.
4. **Phase 7: Live VLM Correlation**
   Linking the network IDS with drone video feeds for a "Second Opinion" verification.

---

## 📁 Repository Structure
- `data_pipeline.py`: Parses raw CSVs to a clean Parquet dataset.
- `train_model.py`: Trains the XGBoost core with full cross-validation metrics.
- `pcap_processor.py`: Scapy-based streaming PCAP feature extractor.
- `inference_engine.py`: Fast model wrapper for single/batch prediction.
- `run_pipeline.py`: **End-to-end runner** (PCAP → Features → IDS → XAI → Arabic/English Briefing).
- `config/`: Central paths and feature definitions.
- `models/`: The trained `model.pkl` artifact.
- `modules/`:
  - `ids_engine.py`: Threat aggregation.
  - `xai_engine.py`: SHAP Truth Trigger extraction.
  - `adversarial_engine.py`: FGSM generation for robustness testing.
  - `tactical_briefing.py`: Bilingual report generator.
  - `data_loader.py`: Dataset connector with simulation fallback.

## 🛠️ Quick Start

**Extract & Train (Optional):**
```bash
python3 data_pipeline.py
python3 train_model.py
```

**Run End-to-End Inference from PCAP:**
```bash
python3 run_pipeline.py data/extracted/ISOT_Drone_Dataset/Raw_Data/DoS/Dos_1_4_30_20mins.pcap
```

See [RESULTS.md](RESULTS.md) for detailed performance metrics.
