# 📡 Proactive Handover Prediction in 5G Networks using Generative Diffusion Models

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![5G NR](https://img.shields.io/badge/Domain-5G%20NR%20%7C%20O--RAN-005A9C?style=flat-square)](https://www.3gpp.org/)
[![Thesis](https://img.shields.io/badge/Degree-M.Tech%20Thesis-success?style=flat-square)](thesis_rishi_v4.pdf)

> **High-Speed Generative Diffusion and DDIM Deterministic Sampling for Risk-Aware Handover in 5G Networks**  
> *Author:* Rishi Thakur (Roll No: 243010109), Dept. of Data Science and Artificial Intelligence, IIIT Naya Raipur  
> *Supervisors:* Dr. Srinivasa KG (Professor & Dean Academics) & Dr. Mallikharjuna Rao K. (Assistant Professor)

---

## 📌 Executive Summary

Mobility management in next-generation cellular networks (5G NR mmWave and 6G) faces acute physical challenges. High-frequency signals suffer from severe free-space path loss and atmospheric attenuation, restricting coverage to dense small cells where line-of-sight (LoS) links are frequently blocked by buildings, vehicles, and foliage. In these environments, **Reference Signal Received Power (RSRP)** can plummet by **20–30 dB in milliseconds**.

Traditional **3GPP reactive handover mechanisms** (such as Event A3) rely on hysteresis margins and a Time-To-Trigger (TTT) window (typically 80–5120 ms). By the time the TTT timer expires and a measurement report is processed, the link has often already severed, resulting in **Radio Link Failure (RLF)** and connection drops.

This repository implements a **proactive, risk-aware generative forecasting framework**:
1. **Conditional Denoising Diffusion Probabilistic Models (DDPM):** Models the full multi-modal distribution of prospective RSRP trajectories given past signal history.
2. **Accelerated Deterministic DDIM Sampling:** Employs non-Markovian step skipping to reduce sampling steps from 100 to 50 (or fewer) without retraining, unlocking near real-time inference suitable for edge deployment.
3. **Autoregressive Diffusion (TimeGrad):** Compares parallel sequence generation against step-by-step autoregressive context expansion.
4. **Three-Tier Handover Decision Logic:** Directly maps forecast uncertainty and failure probabilities into proactive network control actions.

---

## 🏗️ System Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │         5G UE / gNB Telemetry Stream         │
                          │   Past RSRP History (e.g., 50 ms window)     │
                          └──────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────┐
                          │         Context Encoder (GRU Backbone)       │
                          │        Extracts temporal signal dynamics     │
                          └──────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
        ┌────────────────────────────────────────────────────────────────────────────────┐
        │                        Generative Denoising Backbone                           │
        │      Sinusoidal Timestep Embeddings + 4 Residual Blocks (SiLU + Dropout)       │
        └───────────────────┬────────────────────────────────────────┬───────────────────┘
                            │                                        │
                            ▼                                        ▼
             ┌─────────────────────────────┐          ┌─────────────────────────────┐
             │       DDPM Sampler          │          │        DDIM Sampler         │
             │   100-step Stochastic Loop  │          │   50-step Deterministic     │
             │   Full distributional scan  │          │   Fast, real-time edge run  │
             └──────────────┬──────────────┘          └──────────────┬──────────────┘
                            │                                        │
                            └────────────────────┬───────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────┐
                          │   50 Probabilistic Monte Carlo Trajectories  │
                          │        Forecast Horizon: 10 ms / 50 ms       │
                          └──────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────┐
                          │        Risk Engine: P(RSRP < Threshold)      │
                          ├──────────────────────────────────────────────┤
                          │  P > 80%  ➔ 🔴 TRIGGER HANDOVER NOW         │
                          │  P > 40%  ➔ 🟡 PREPARE HANDOVER / MEAS GAP  │
                          │  P ≤ 40%  ➔ 🟢 STAY CONNECTED               │
                          └──────────────────────────────────────────────┘
```

---

## 🚀 Key Innovations & Model Variants

### 1. Base Parallel Diffusion Model (`diffusion_model.py`)
- **Architecture:** `RSRPDiffusion`
- **Context Encoder:** Gated Recurrent Unit (`nn.GRU`, `context_dim=64`) encoding sequential history into a conditioning vector.
- **Denoising Core:** 4-layer residual blocks (`hidden_dim=128`) featuring SiLU activations and Dropout.
- **Timestep Embedding:** Continuous sinusoidal position embeddings projected through a 2-layer MLP.
- **Inference Mode:** Parallel denoising across the entire prediction horizon simultaneously.

### 2. Deterministic DDIM Sampler (`ddim_sampler.py`, `ddim.py`)
- Leverages non-Markovian forward diffusion with identical marginal distributions.
- When $\eta = 0.0$, reverse sampling becomes completely deterministic.
- Enables **50-step sub-sequence evaluation** from weights trained on 100 timesteps—achieving significant latency reduction with no loss in accuracy.

### 3. Autoregressive Diffusion Model (`timegrad_model.py`)
- Implements the TimeGrad formulation for time series.
- Single forward pass training with teacher forcing.
- Sequential autoregressive rollouts: each future point is denoised via reverse diffusion and fed back into the GRU state before generating subsequent points.

---

## 📊 Experimental Results & Benchmarks

The models were evaluated on real-world 5G drive test measurement traces across 10 ms and 50 ms prediction horizons.

### Quantitative Comparison (`ddpm_ddim_metrics.csv` & Benchmark Runs)

| Metric | DDPM (100 steps) | DDIM (50 steps) | TimeGrad (Autoregressive) |
|---|:---:|:---:|:---:|
| **Sampling Mechanism** | Stochastic (Markovian) | Deterministic (Non-Markovian) | Autoregressive (Recurrent) |
| **Inference Steps** | 100 | 50 | $10 \times 100 = 1000$ |
| **MAE (dB)** | **3.751 ± 2.18** | **3.795 ± 2.16** | 3.820 ± 2.25 |
| **CRPS Score** | **2.750 ± 1.41** | **2.843 ± 1.25** | 2.890 ± 1.35 |
| **Recall / Sensitivity** | **85.7%** | **85.7%** | 84.2% |
| **Average Latency** | ~355 ms | **~203 ms (~1.75× speedup)** | ~2800 ms |
| **Edge Feasibility** | Moderate | **High (Sub-sequence ready)** | Offline / Analysis only |

> 💡 **Core Finding:** DDIM matches the prediction fidelity (MAE and CRPS) and critical failure detection rate of 100-step DDPM while cutting computation in half without requiring model retraining.

---

## 📁 Repository Structure

```
.
├── calculate_advanced_metric.py    # Generates 7 publication figures + comprehensive metrics
├── compare_all_models.py           # Benchmark script comparing DDPM, DDIM, and TimeGrad
├── dataset_loader.py               # RSRP sliding-window PyTorch Dataset & DataLoader
├── ddim.py                         # Standalone DDPM & DDIM mathematical functions
├── ddim_sampler.py                 # Core shared sampling library (DDPM, DDIM, TimeGrad)
├── ddim_visualization.py           # Visualization of DDIM probabilistic clouds
├── diffusion_model.py              # Main parallel RSRPDiffusion neural network
├── timegrad_model.py               # Autoregressive TimeGradModel neural network
├── train.py                        # Unified training script for DDPM & TimeGrad (pred_len 10/50)
├── train_diffusion.py              # Base training script for RSRPDiffusion
├── run_handover.py                 # Real-time 3-tier risk-aware handover decision engine
├── scan_for_handover.py            # Sequential event scanner detecting weak signal regions
├── test_overfitting.py             # Generalization check: Train MSE vs. Test MSE
├── visualize_results.py            # Side-by-side & overlay trajectory comparison plots
├── diffusion_handover_model.pth    # Trained PyTorch checkpoint (hidden_dim=128)
├── ddpm_ddim_metrics.csv           # Benchmark metric output data
├── thesis_rishi_v4.pdf             # Complete M.Tech Thesis document
├── research-paper-1.pdf            # Related publication / reference paper
├── research-paper-2.pdf            # Related publication / reference paper
└── README.md                       # Repository documentation
```

---

## 🛠️ Getting Started

### 1. Prerequisites & Installation

Create and activate a virtual environment, then install dependencies:

```bash
git clone https://github.com/rishi-th219/Proactive-Handover-Prediction-in-5G-using-Diffusion-Models.git
cd Proactive-Handover-Prediction-in-5G-using-Diffusion-Models

python3 -m venv venv
source venv/bin/activate

pip install torch numpy pandas matplotlib
```

### 2. Dataset Setup
Place your drive-test CSV files in the `data/` directory:
- `data/drive_test_measurements01.csv` (Training trace)
- `data/drive_test_measurements03.csv` (Testing/validation trace)

*(Each CSV must contain an `RSRP` column with signal measurements in dBm).*

---

## 💻 Running the Code

### 🏃 1. Train Models
To train the unified models across different architectures and horizons:
```bash
python train.py
```
*(Configure `model_type` (`"ddpm"` or `"timegrad"`) and `pred_len` (`10` or `50`) in the `CONFIG` dictionary at the top of `train.py`).*

### 🧪 2. Run Comprehensive Model Benchmarking
Compare DDPM-10, DDIM-10, TimeGrad-10, DDPM-50, and TimeGrad-50:
```bash
python compare_all_models.py
```
Outputs tabular comparisons and bar charts in `results/`.

### 📈 3. Generate Publication Figures
Compute CRPS decomposition, error distributions, calibration curves, and temporal stability:
```bash
python calculate_advanced_metric.py
```
Generates 7 high-resolution figures (`fig1_metrics_bar.png` through `fig7_temporal_stability.png`).

### 🚨 4. Simulate Real-Time Handover Decisions
Run the probabilistic risk engine on a live signal window:
```bash
python run_handover.py
```
Example Output:
```text
Model     : DDIM
Checkpoint: diffusion_handover_model.pth

Current RSRP: -94.20 dBm
Generating 50 probabilistic futures (10 ms ahead)...

Inference time : 0.0820s
Min predicted  : -112.40 dBm
Mean predicted : -98.10 dBm
Risk (P < -85 dBm): 88.0%

───────────────────────────────────
HANDOVER DECISION
───────────────────────────────────
🔴  TRIGGER HANDOVER IMMEDIATELY
    Reason: >80% of futures predict signal failure.
───────────────────────────────────
```

### 🔍 5. Scan Traces for Critical Handover Events
Sequentially scan a drive test log and trigger accelerated DDIM sampling when weak signal conditions arise:
```bash
python scan_for_handover.py
```

### 🎨 6. Visualize Trajectory Ensembles
Generate side-by-side forecast clouds and mean trajectory comparisons:
```bash
python visualize_results.py
```

---

## 📜 Thesis Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@mastersthesis{thakur2026diffusionhandover,
  author       = {Rishi Thakur},
  title        = {High-Speed Generative Diffusion and DDIM Deterministic Sampling for Risk-Aware Handover in 5G Networks},
  school       = {Dr. SPM International Institute of Information Technology, Naya Raipur},
  year         = {2026},
  month        = {June},
  address      = {Naya Raipur, Chhattisgarh, India},
  note         = {Supervised by Dr. Srinivasa KG and Dr. Mallikharjuna Rao K.}
}
```

---

## 📄 License
This project is released under the [MIT License](LICENSE) for academic and research purposes.
