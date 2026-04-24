# 📡 Proactive Handover Prediction in 5G using Diffusion Models

This project tackles a key limitation in 5G mobility management: the reliance on **reactive handover mechanisms**. Instead of waiting for signal degradation events, this system introduces a **proactive, AI-driven framework** that predicts future signal conditions and enables early, risk-aware handovers.

---

## 📌 Problem Statement

Current 5G handover logic is based on **Event A3**, which triggers only after:
- A signal threshold is crossed  
- Maintained over a **Time-To-Trigger (TTT)** interval  

In **mmWave environments**, signal drops of **20–30 dB can occur within milliseconds**, making reactive strategies too slow and resulting in:
- ❌ Radio Link Failures (RLF)  
- ❌ Increased latency  
- ❌ Reduced reliability  

---

## 🚀 Key Features

### 🔮 Generative Signal Forecasting
- Implements **Conditional Denoising Diffusion Probabilistic Models (DDPM)**
- Generates **50+ plausible future RSRP trajectories**
- Captures uncertainty in dynamic wireless environments

### ⚠️ Risk-Aware Decision Making
- Uses **Continuous Ranked Probability Score (CRPS)**
- Enables probabilistic handover decisions:
- Trigger HO if P(Failure) < 5%
- 
### 🎯 Dual-Masking Mechanism
- Focuses learning on:
- Candidate cells
- Critical protocol time points (TTT, MTS)
- Improves operational accuracy

### 🧠 Hybrid Two-Stage Framework
1. **Diffusion-based regression model**
2. **Random Forest classifier** for final handover decision

---

## 🏗️ Technical Architecture

The system uses a **conditional diffusion framework for time-series RSRP prediction**.

### 1️⃣ Model Components (`diffusion_model.py`)

- **Context Encoder**
- GRU-based RNN
- Encodes historical signal data

- **Denoising Network**
- Residual blocks with:
  - SiLU activation
  - Dropout regularization

- **Time Embedding**
- MLP encoding diffusion timestep

---

### 2️⃣ Training & Inference

#### Forward Diffusion
- Adds Gaussian noise over **100 timesteps**

#### Reverse Diffusion
- Iteratively denoises from pure noise
- Generates realistic future signal trajectories

---

## 📊 Performance Evaluation

| Metric | LSTM | SegRNN | **Diffusion Model** |
|--------|------|--------|---------------------|
| Model Type | Deterministic | Deterministic | **Probabilistic** |
| MAE | 0.6 dB | 1.26 dB | 1.80 dB |
| F1 Score | 97.6% | 94.6% | **97.64%** |
| CRPS | N/A | N/A | **0.044** |
| Uncertainty | ❌ | ❌ | ✅ |

> ⚠️ **Note:**  
> Although MAE is slightly higher, diffusion models are critical for **URLLC scenarios** due to their ability to model uncertainty and rare events.

---

## 🌐 O-RAN Deployment

Designed for integration into the **O-RAN Intelligent Controller (RIC)**:

### 📍 Near-RT RIC (xApp)
- Real-time inference (10 ms – 1 s loop)
- Predicts RSRP & triggers handovers

### 📍 Non-RT RIC (rApp)
- Model training & updates
- Data aggregation
- Policy distribution via A1 interface

---
## ✅ ConclusionThis project demonstrates that:- **Reactive handovers are insufficient** for modern 5G scenarios  - **Diffusion models enable proactive decision-making**  - **Uncertainty-aware systems improve reliability** in URLLC environments  ---## 📬 Future Work- Real-time deployment validation on live networks  - Latency optimization for edge inference  - Integration with multi-cell coordination strategies  
