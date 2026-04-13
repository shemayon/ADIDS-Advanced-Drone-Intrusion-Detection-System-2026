# A-DIDS: AI-Driven Intrusion Detection for Drone Networks
> **Built a real-time intrusion detection system for drone networks, converting live packet streams into explainable ML predictions with millisecond latency.**

*A Production-Ready ML Systems Framework for Tactical Edge UAV Telemetry*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

## 🚀 Project Highlights
- **Real-time packet streaming** → ML inference pipeline.
- **Multi-class intrusion detection** (DoS, MITM, Spoofing, etc.).
- **Explainable AI** using SHAP (feature-level reasoning).
- **FastAPI deployment** for seamless production integration.
- **Validated on mixed traffic** with low False Positive Rate (1.07%).

## ⚙️ Tech Stack
- **Languages/Tools**: Python, Scapy (Live Sniffing)
- **ML Core**: XGBoost, Isolation Forest (Zero-Day)
- **Explainability**: SHAP (Truth Triggers)
- **Deployment**: FastAPI, Uvicorn, REST API
- **Data**: Network Flow Analysis (Bidirectional 5-tuple)

---

## 🏗️ System Architecture

A-DIDS utilizes a modular pipeline designed for real-world deployment, merging low-level data engineering with explainable AI.

```mermaid
graph TD
    A[Live Traffic / PCAP] --> B[Scapy Live Sniffer]
    B --> C[Real-Time Flow Reconstruction]
    C --> D[XGBoost Detection Core]
    D --> E[FastAPI Microservice]
    D --> F[SHAP Explainability]
    E --> G[Tactical Command Dashboard]
    
    subgraph "Production Defense Layers"
    D --> H[Zero-Day Anomaly Detection]
    D --> I[SDN Mitigation (simulated)]
    end
```

## ⚡ Quick Start

```bash
# 1. Install Production Dependencies
pip install -r requirements.txt

# 2. Preprocess Dataset (Converts raw CSVs to Parquet)
python data_pipeline.py --base-path "/path/to/extracted/dataset"

# 3. Train the Multi-Class Model
python train_multiclass.py

# 4. Start Live Sniffer (Requires Root for Capture)
sudo python live_sniffer.py

# 5. Launch Production API
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

## 📅 Dataset Reference
This project utilizes the **ISOT Drone Dataset**, a specialized collection of Wi-Fi-based telemetry from the **DJI Edu Tello** drone. It covers a wide spectrum of adversarial scenarios including DoS, MITM, and Injection attacks.

- **Download Link**: [ISOT Drone Dataset (Google Drive)](https://drive.google.com/file/d/1ILTxl5p4j4zBnQNuEOTnyyviZEgHpDtF/view)
- **Citation**: *ISOT Drone Dataset - Wi-Fi-based DJI Edu Tello drone network flow collection.*

---

## 🚀 Real-Time Production Performance

### 🧪 Evaluation Setup
- **Dataset**: ISOT Drone Dataset (~2.9M samples).
- **Traffic**: Randomized mixed benign + attack streams.
- **Environment**: x86 VM (4 vCPU, 16GB RAM).
- **Method**: Flow-based feature inference (19 features).

### System Metrics (Production Stress Test)
*Evaluated on a mixed stream of 100,000 baseline and attack flows.*

| Metric | Performance | Description |
|:---|:---|:---|
| **Average Latency** | **0.0022 ms** | Inference time per flow (XGBoost) |
| **System Throughput** | **454,729 flows/sec** | Estimated throughput under test conditions |
| **False Positive Rate** | **1.07%** | Operational reliability under noise |
| **Recall (Detection)** | **99.52%** | Effectiveness against active threats |

---

## 🛡️ Core Engineering Capabilities

### 1. Live Stream Ingestion (`live_sniffer.py`)
A-DIDS moves beyond file processing. The `live_sniffer.py` module utilizes a sliding-window flow accumulator to reconstruct bidirectional 5-tuple flows from live network interfaces, performing inference the moment a flow finalized.

### 2. Production API Deployment (`api/app.py`)
The system is exposed as a **FastAPI** microservice, allowing seamless integration with ground control stations or onboard flight computers.

#### 🌐 API Usage Example
**POST /predict**
```json
{
  "features": [0.032, 12, 10, 850.5, 92.1, 15.2, 14.8, 16.1, 240, 200, 4.2, 3.8, 0, 1500, 750.2, 750.2, 4200, 8192, 8192]
}
```
**Response**
```json
{
  "prediction": "ATTACK",
  "confidence": 0.9982,
  "latency_ms": 0.0024
}
```

### 📟 Sample Output
```text
2026-04-13 10:14:12 [INFO] Starting Live IDS Sniffer on eth0...
2026-04-13 10:14:45 [INFO] [!!! ATTACK DETECTED !!!] Flow: 192.168.1.50 <-> 192.168.1.100 | Confidence: 99.82% | Latency: 0.0024ms
2026-04-13 10:14:45 [INFO] Truth Triggers: [Payload Entropy: 7.2, Packet Rate: 1.5k/s]
```

### 3. Adversarial Hardening (FGSM)
To protect against "Stealth AI" evasions, the system incorporates an **Adversarial Engine** using the Fast Gradient Sign Method (FGSM) to harden the XGBoost core against malicious feature perturbations.

---

## ⚠️ Limitations
- **Environment**: Live testing performed in a controlled network environment; wide-area field tests needed.
- **Hardware Integration**: RF/SDR modules are currently algorithmic simulations (no SDR hardware integration yet).
- **Edge Deployment**: Performance benchmarks are server-based; Jetson-specific optimization is still in progress.

---

## 🧪 Simulation Extensions (Tactical Alignment)
To align with the requirements, the framework includes algorithmic simulations for physical-layer and swarm coordination:
*   📡 **RF PHY Layer:** Models the correlation between network anomalies and Physical-Layer signal jamming.
*   🛰️ **Swarm Consensus:** A Byzantine Fault Tolerance simulation for corroborating alerts across multiple peer nodes.

---

## 🛰️ Future Research & Roadmap (2026 Vision)

### 1. Multi-Agent Coordination Layer
Exploring hierarchical agent-based decision layers for autonomous swarm defense, utilizing specialized Perceptor and Reasoning Agents.

### 2. Edge Hardware Optimization
Benchmarking on **NVIDIA Jetson Orin** and **Ettus USRP** platforms to achieve true hardware-in-the-loop (HIL) operational status.

---
*Developed by shemayons for the 2026 Defensive AI Ecosystem.*
