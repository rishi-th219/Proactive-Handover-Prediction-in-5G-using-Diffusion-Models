# 📡 Proactive Handover Prediction in 5G Networks using Generative Diffusion Models

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![5G NR mmWave](https://img.shields.io/badge/Domain-5G%20NR%20%7C%20mmWave%20%7C%20O--RAN-005A9C?style=flat-square)](https://www.3gpp.org/)
[![M.Tech Thesis](https://img.shields.io/badge/Degree-M.Tech%20Thesis-success?style=flat-square)](thesis_rishi_v4.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

> **High-Speed Generative Diffusion and DDIM Deterministic Sampling for Risk-Aware Handover in 5G Networks**  
> **Author:** Rishi Thakur (Roll No: 243010109)  
> **Supervisors:** Dr. Srinivasa KG (Professor & Dean Academics) & Dr. Mallikharjuna Rao K. (Assistant Professor)  
> **Department:** Department of Data Science and Artificial Intelligence  
> **Institution:** Dr. SPM International Institute of Information Technology, Naya Raipur (IIIT-NR), India (June 2026)  
> **Full Thesis:** [`thesis_rishi_v4.pdf`](thesis_rishi_v4.pdf)

---

## 📖 Table of Contents
- [1. The 5G mmWave Mobility Problem](#-1-the-5g-mmwave-mobility-problem)
  - [1.1 Physical Channel Degradation](#11-physical-channel-degradation)
  - [1.2 The Failure of Standard 3GPP Reactive Handover](#12-the-failure-of-standard-3gpp-reactive-handover)
- [2. The Proactive Handover Paradigm](#-2-the-proactive-handover-paradigm)
  - [2.1 Why Point Forecasts (LSTM/DLinear) Fall Short](#21-why-point-forecasts-lstmdlinear-fall-short)
  - [2.2 The Generative Diffusion Solution](#22-the-generative-diffusion-solution)
- [3. System Architecture & Methodology](#-3-system-architecture--methodology)
  - [3.1 Model Architecture: `RSRPDiffusion`](#31-model-architecture-rsrpdiffusion)
  - [3.2 Accelerated Inference via DDIM Deterministic Sampling](#32-accelerated-inference-via-ddim-deterministic-sampling)
  - [3.3 Autoregressive Alternative: TimeGrad](#33-autoregressive-alternative-timegrad)
  - [3.4 Real-Time 3-Tier Risk Decision Engine](#34-real-time-3-tier-risk-decision-engine)
- [4. Experimental Setup & Dataset](#-4-experimental-setup--dataset)
  - [4.1 Real-World Drive-Test Campaign (Belo Horizonte)](#41-real-world-drive-test-campaign-belo-horizonte)
  - [4.2 Preprocessing and Sliding-Window Protocol](#42-preprocessing-and-sliding-window-protocol)
- [5. Quantitative Benchmarks & Research Findings](#-5-quantitative-benchmarks--research-findings)
  - [5.1 Performance Comparison Table](#51-performance-comparison-table)
  - [5.2 The Safety Advantage: Why DDIM Cuts Missed Triggers](#52-the-safety-advantage-why-ddim-cuts-missed-triggers)
  - [5.3 Speedup vs. Quality Trade-Off](#53-speedup-vs-quality-trade-off)
- [6. O-RAN & SDN Deployment Architecture](#-6-o-ran--sdn-deployment-architecture)
- [7. Codebase Guide & Repository Structure](#-7-codebase-guide--repository-structure)
- [8. Quickstart & Usage Instructions](#-8-quickstart--usage-instructions)
- [9. Citation](#-9-citation)

---

## 📡 1. The 5G mmWave Mobility Problem

### 1.1 Physical Channel Degradation
Fifth-generation (5G) New Radio (NR) and emerging 6G wireless networks heavily utilize **millimeter-wave (mmWave)** (24–52 GHz, FR2) and terahertz (THz) spectrum to support multi-gigabit throughput and Ultra-Reliable Low-Latency Communication (URLLC). However, signal propagation at these frequencies obeys Friis' free-space transmission law where path loss scales quadratically with frequency:

$$\text{PL}(d, f) = 20\log_{10}(d) + 20\log_{10}(f) + 20\log_{10}\left(\frac{4\pi}{c}\right)$$

Due to high atmospheric absorption, severe rain attenuation, and negligible diffraction around obstacles:
- **Concrete walls** introduce **35–50 dB** of attenuation.
- **Tinted glass** introduces **20–30 dB** of loss.
- **Physical blockages** (vehicles, pedestrians, street corners) instantly sever the Line-of-Sight (LoS) path.

As User Equipment (UE) moves through urban environments at vehicular speeds (50–60 km/h), the **Reference Signal Received Power (RSRP)** can drop precipitously by **20–30 dB within a few milliseconds**.

```
                ┌──────────────────────────────────────────────────────────────┐
                │             The mmWave Severe Blockage Challenge            │
                └──────────────────────────────┬───────────────────────────────┘
                                               │
               LoS Path (Healthy Signal)       │       Obstacle / Shadow Fading
               RSRP ≈ -65 to -75 dBm           │       RSRP drops by 20-30 dB in <10ms
                                               ▼
                ┌──────────────────────────────────────────────────────────────┐
                │ Outage Zone (< -95 dBm) ➔ Exponential BLER ➔ Radio Link Fail │
                └──────────────────────────────────────────────────────────────┘
```

---

### 1.2 The Failure of Standard 3GPP Reactive Handover
Standard cellular networks rely on **reactive event-triggered protocols**, primarily **3GPP Event A3**:

$$\text{RSRP}_{\text{target}} > \text{RSRP}_{\text{serving}} + \text{Hysteresis}$$

To prevent noisy channel fluctuations from triggering unnecessary handovers (ping-pong effect), standard 3GPP requires this condition to persist continuously for a **Time-To-Trigger (TTT)** interval (configured between **80 ms and 5120 ms**).

* **The Core Bottleneck:** In dense mmWave small cells, the RSRP decays to link failure levels in **under 20 ms**. By the time the TTT timer expires and a Measurement Report is received, the serving link has already collapsed into a **Radio Link Failure (RLF)**, dropping user sessions, interrupting URLLC services, and congesting control planes.

---

## 🔮 2. The Proactive Handover Paradigm

### 2.1 Why Point Forecasts (LSTM/DLinear) Fall Short
Prior machine learning approaches (LSTM, SegRNN, DLinear) formulate handover forecasting as a **deterministic regression** task: predicting a single future RSRP trajectory $\hat{Y} \in \mathbb{R}^{W_{\text{out}} \times 1}$.

However, high-frequency wireless channels are governed by severe **aleatoric uncertainty** (multipath interference, fast fading, dynamic vehicular scatterers):
1. **Mean Collapse:** Point-forecast models trained on MSE regress to the empirical conditional mean, which smooths out sharp, catastrophic drops.
2. **Binary Risk Blindness:** If a deterministic model predicts $-83\text{ dBm}$ (just above a $-85\text{ dBm}$ outage threshold), the controller takes no action—even if there is a 40% probability of dropping to $-92\text{ dBm}$.

### 2.2 The Generative Diffusion Solution
This project reformulates proactive handover as a **conditional generative time-series modeling problem**. Given past signal history $X_{\text{hist}}$, our model samples the true underlying posterior distribution:

$$P\left(Y_{\text{future}} \mid X_{\text{hist}}\right)$$

By generating an ensemble of $N = 50$ probabilistic trajectories, the network controller gains full visibility into tail-risk failure probabilities.

---

## 🏗️ 3. System Architecture & Methodology

```
                                  PROACTIVE HANDOVER SYSTEM OVERVIEW
                                  
  ┌─────────────────────────┐
  │ 50 ms RSRP Past History │ ───┐
  │   x_hist ∈ ℝ^(50×1)     │    │
  └─────────────────────────┘    │
                                 ▼
                    ┌─────────────────────────┐
                    │    Context GRU Encoder  │ (hidden_dim = 64)
                    │  Extracts Macro-Trends  │
                    └────────────┬────────────┘
                                 │ Context Vector c ∈ ℝ^128
                                 ▼
  ┌─────────────────────────┐   ┌───────────────────────────┐
  │ Timestep t ∈ [1, 100]   │──▶│ Sinusoidal Pos. Embedding │──▶ t_emb ∈ ℝ^128
  └─────────────────────────┘   └───────────────────────────┘
                                 │
  ┌─────────────────────────┐   ┌───────────────────────────┐
  │ Noisy Target x_noisy    │──▶│ Input Linear Projection   │──▶ x_proj ∈ ℝ^128
  └─────────────────────────┘   └───────────────────────────┘
                                 │
                                 ▼
                     h_0 = x_proj + t_emb + c (Element-wise Addition)
                                 │
                                 ▼
                    ┌───────────────────────────┐
                    │    4x Residual Blocks     │ (Linear + SiLU + Linear + Dropout 0.1)
                    │   Skip-connection Core    │
                    └────────────┬──────────────┘
                                 │
                                 ▼
                    ┌───────────────────────────┐
                    │     Output Projection     │──▶ Predicted Noise ϵ̂ ∈ ℝ^(10×1)
                    └───────────────────────────┘
                                 │
        ┌────────────────────────┴────────────────────────┐
        ▼                                                 ▼
┌───────────────────────────────┐         ┌───────────────────────────────┐
│     Standard DDPM Loop        │         │     Accelerated DDIM Loop     │
│   100 Stochastic Steps (η=1)  │         │   50 Deterministic Steps (η=0)│
│   Latency: ~355 ms / window   │         │   Latency: ~188 ms / window   │
│   Stochastic Trajectories     │         │   1.97× Speedup, Lower FNR    │
└───────────────┬───────────────┘         └───────────────┬───────────────┘
                │                                         │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │ 50 Probabilistic Trajectories Ensemble  │
                └────────────────────┬────────────────────┘
                                     │
                                     ▼
                ┌─────────────────────────────────────────┐
                │  Risk Evaluator: P(min(RSRP) < -85 dBm) │
                └────────────────────┬────────────────────┘
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
 P(Fail) ≤ 40%              40% < P(Fail) ≤ 80%              P(Fail) > 80%
🟢 STAY CONNECTED           🟡 PREPARE HANDOVER              🔴 TRIGGER HANDOVER
 Link Predicted Healthy      Configure Measurement Gap       Preemptive Link Switch
```

---

### 3.1 Model Architecture: `RSRPDiffusion`
Defined in [`diffusion_model.py`](diffusion_model.py):
1. **Context GRU Encoder:** A single-layer Gated Recurrent Unit (`hidden_dim = 64`) processes 50 ms of historical RSRP, extracting $h_d[-1] \in \mathbb{R}^{64}$. It captures macro-mobility trendlines while filtering high-frequency multipath noise.
2. **Sinusoidal Timestep Embedding:** The diffusion timestep $t \in [1, 100]$ is embedded into dimension $D = 128$ using harmonic frequencies:
   $$\text{Embed}(t)_{2i} = \sin\left(t \cdot 10000^{-2i/D}\right), \quad \text{Embed}(t)_{2i+1} = \cos\left(t \cdot 10000^{-2i/D}\right)$$
   Projected through a two-layer SiLU MLP (`Linear(128, 128) -> SiLU -> Linear(128, 128)`).
3. **Element-Wise Additive Fusion:** Input projection $x_{\text{proj}}$, context projection $c_{\text{proj}}$, and timestep embedding $t_{\text{emb}}$ are combined via element-wise addition:
   $$h_0 = x_{\text{proj}} + t_{\text{emb}} + c_{\text{proj}}$$
   *(Keeps internal dimensions at 128, minimizing parameter footprint for edge inference).*
4. **4x Residual Processing Blocks:** Deep residual blocks with skip connections:
   $$h_{l+1} = h_l + \text{Dropout}\left(\text{Linear}\left(\text{SiLU}\left(\text{Linear}(h_l)\right)\right)\right)$$
5. **Parallel Denoising Target:** Predicts the noise tensor for all future steps $W_{\text{out}} = 10$ concurrently, avoiding autoregressive inference slowdowns.

---

### 3.2 Accelerated Inference via DDIM Deterministic Sampling
Defined in [`ddim_sampler.py`](ddim_sampler.py) and [`ddim.py`](ddim.py):

Standard DDPM requires $T = 100$ sequential stochastic transitions. **Denoising Diffusion Implicit Models (DDIM)** reformulate the forward process into a non-Markovian family with matching marginal distributions, allowing deterministic reverse stepping ($\eta = 0.0$):

$$\hat{x}_0 = \text{clamp}\left(\frac{x_t - \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t, h)}{\sqrt{\bar{\alpha}_t}}, -1.0, 1.0\right)$$

$$x_{\text{prev}} = \sqrt{\bar{\alpha}_{\text{prev}}} \cdot \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{\text{prev}}} \cdot \epsilon_\theta(x_t, t, h)$$

#### 🛡️ Crucial Physical Clamping
In unconstrained diffusion, large initial noise can cause predictions of $\hat{x}_0$ to exceed $[-1, 1]$, which compounds across steps and causes gradient explosion. Clamping $\hat{x}_0$ to $[-1.0, 1.0]$ enforces the physical bounds of the radio receiver ($-101.0\text{ dBm}$ to $-45.5\text{ dBm}$), guaranteeing numerical stability.

#### ⚡ Sub-Sequence Step Scheduling
Instead of evaluating all 100 timesteps, DDIM samples a linear sub-sequence of length $S = 50$:
$$\tau = [99, 97, 95, \dots, 1]$$
- **Result:** Neural network passes are cut by **50% (2× speedup)**.
- **Zero Retraining:** Operates directly on the weights of the trained DDPM model (`diffusion_handover_model.pth`).

---

### 3.3 Autoregressive Alternative: TimeGrad
Defined in [`timegrad_model.py`](timegrad_model.py):
- Unlike parallel DDPM, TimeGrad predicts future points sequentially.
- **Training:** Uses teacher forcing over concatenated $[X_{\text{hist}} \parallel X_{\text{future\_noisy}}]$ in a single GRU pass.
- **Inference:** Denoises step $k$, feeds the generated scalar into the GRU hidden state, and denoises step $k+1$.
- **Trade-off:** For $W_{\text{out}} = 10$, TimeGrad executes $10 \times 100 = 1000$ denoising passes, resulting in ~2.8s latency (offline evaluation only).

---

### 3.4 Real-Time 3-Tier Risk Decision Engine
Implemented in [`run_handover.py`](run_handover.py) & [`scan_for_handover.py`](scan_for_handover.py):

In commercial networks, **$\gamma = -85\text{ dBm}$** marks the edge-coverage boundary where Block Error Rate (BLER) surges. If signal drops below $-95\text{ dBm}$, complete Radio Link Failure occurs.

For each generated trajectory $Y^{(s)}$ ($s = 1, \dots, N=50$):

$$I^{(s)} = \mathbb{I}\left(\min_{k=1\dots W_{\text{out}}} y_{t+k}^{(s)} < \gamma\right)$$

$$P_{\text{failure}} = \frac{1}{N} \sum_{s=1}^N I^{(s)}$$

| Risk Probability ($P_{\text{failure}}$) | Control Decision | Network Action |
|:---:|:---:|:---|
| **$P \le 0.40$** | 🟢 **STAY CONNECTED** | Serving link stable; no control overhead. |
| **$0.40 < P \le 0.80$** | 🟡 **PREPARE HANDOVER** | Moderate risk; configure measurement gaps and pre-allocate target cell resources. |
| **$P > 0.80$** | 🔴 **TRIGGER HANDOVER IMMEDIATELY** | Critical risk; execute proactive handover immediately, bypassing 3GPP Time-To-Trigger (TTT). |

---

## 📊 4. Experimental Setup & Dataset

### 4.1 Real-World Drive-Test Campaign (Belo Horizonte)
The models were trained and benchmarked using commercial LTE drive-test measurements collected in downtown **Belo Horizonte, Brazil** (the established benchmark dataset used by Lima et al.):
- **Equipment:** Rohde & Schwarz TSMW radio network analyzer mounted on an urban test vehicle.
- **Urban Environment:** Deep urban building canyons, high-density traffic, varied road elevations, and severe multipath/shadow fading.
- **Vehicle Speed:** ~50 km/h.
- **Sampling Frequency:** High-resolution 1 ms sampling interval.
- **Data Partitions:**
  - `drive_test_measurements01.csv`: 120,000 samples (Training split)
  - `drive_test_measurements02.csv`: 120,000 samples (Training split)
  - `drive_test_measurements03.csv`: 60,000 samples (Held-out, strictly unseen test split)

### 4.2 Preprocessing and Sliding-Window Protocol
- **Normalization:** Min-Max normalized to $[-1, 1]$ based on empirical extremities ($RSRP_{\min} = -101.0\text{ dBm}$, $RSRP_{\max} = -45.5\text{ dBm}$).
- **Sliding Window:**
  - History window ($W_{\text{in}}$): 50 ms (50 observations)
  - Prediction horizon ($W_{\text{out}}$): 10 ms (short horizon, matching 5G radio frame duration) and 50 ms (long horizon).

---

## 📈 5. Quantitative Benchmarks & Research Findings

### 5.1 Performance Comparison Table
Evaluated across 100 sliding windows on the held-out test dataset (`drive_test_measurements03.csv`):

| Model Configuration | Sampling Method | Denoising Steps | MAE (dB) | CRPS Score | False Negative Rate (FNR) | Latency / Window | Speedup Factor |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **DDPM-10 (Baseline)** | Stochastic | 100 | **3.92 dB** | **2.82** | 21.43% | 0.372 s | 1.00× (Baseline) |
| **DDIM-10 (Ours)** | **Deterministic** | **50** | **4.03 dB** | **2.86** | **14.29%** | **0.188 s** | **1.97× Faster** |
| **TimeGrad-10** | Autoregressive | 1000 | 3.82 dB | 2.89 | 20.00% | 2.810 s | 0.13× (Very Slow) |
| **DDPM-50** | Stochastic | 100 | 4.88 dB | 3.45 | 28.57% | 0.395 s | 1.00× (Baseline) |
| **TimeGrad-50** | Autoregressive | 5000 | 4.61 dB | 3.38 | 25.00% | 14.20 s | 0.03× (Impractical) |

---

### 5.2 The Safety Advantage: Why DDIM Cuts Missed Triggers
A standout finding of this thesis is that **DDIM reduces the False Negative Rate by 7.14 percentage points (from 21.43% down to 14.29%)** compared to baseline DDPM:
- **DDPM Stochastic Flaw:** Standard DDPM injects random Gaussian noise $\sigma_t z_t$ at every reverse step. In near-threshold scenarios, random noise can generate unrealistically optimistic trajectory realizations that fail to cross $\gamma = -85\text{ dBm}$, causing the system to miss critical handover triggers.
- **DDIM Deterministic Stability:** Setting $\eta = 0.0$ removes random noise injection during reverse traversal. Diversity is driven strictly by initial noise latents $x_T \sim \mathcal{N}(0, I)$ mapped through a consistent trajectory, producing a stable distribution that reliably captures tail risks and severe blockages.

---

### 5.3 Speedup vs. Quality Trade-Off
- **Latency Reduction:** DDIM executes in **0.188 s per window** (compared to 0.372 s for DDPM), achieving a **1.97× speedup**.
- **Negligible Quality Penalty:** The MAE difference between 100-step DDPM and 50-step DDIM is **only 0.11 dB**, and the Continuous Ranked Probability Score (CRPS) differs by **only 0.04**.
- **Conclusion:** Halving the step count yields double the execution speed while preserving forecast fidelity and improving handover safety.

---

## 🌐 6. O-RAN & SDN Deployment Architecture

This framework aligns directly with **Open RAN (O-RAN)** standards:

```
                            O-RAN SERVICE MANAGEMENT & ORCHESTRATION (SMO)
                                                   │
                            ┌──────────────────────┴──────────────────────┐
                            ▼                                             ▼
                 Non-RT RIC (rApp)                             Near-RT RIC (xApp)
          • Offline Training on Drive Logs             • Online DDIM-10 Deterministic Inference
          • Hyperparameter Tuning & Updates            • Sub-200 ms Execution Budget
          • Pushes Model Weights via A1                • Evaluates P(Failure) & Triggers Handover
                            │                                             │
                            └──────────────────────┬──────────────────────┘
                                                   │ E2 Interface / OpenFlow
                                                   ▼
                                         5G gNodeB / Data Plane
                                    • Telemetry Ingestion (1 ms RSRP)
                                    • Instant Flow Re-Routing
```

- **Near-RT RIC xApp:** Hosts the compiled PyTorch DDIM engine. Processes rolling 50 ms telemetry windows, generates 50 futures in ~188 ms, and issues OpenFlow modifications to re-route bearer traffic prior to signal collapse.
- **Non-RT RIC rApp:** Aggregates regional telemetry, executes background fine-tuning via `train.py`, and distributes updated checkpoints to edge base stations.

---

## 📁 7. Codebase Guide & Repository Structure

| File | Category | Description |
|---|---|---|
| [`diffusion_model.py`](diffusion_model.py) | **Architecture** | Main `RSRPDiffusion` network (GRU context encoder + Sinusoidal Embeddings + 4 Residual Blocks). |
| [`timegrad_model.py`](timegrad_model.py) | **Architecture** | Autoregressive `TimeGradModel` network with teacher forcing training and step-by-step rollout. |
| [`ddim_sampler.py`](ddim_sampler.py) | **Sampling** | Unified sampling engine implementing `ddpm_sample`, `ddim_sample`, `timegrad_sample`, and `denorm`. |
| [`ddim.py`](ddim.py) | **Sampling** | Standalone mathematical implementations of DDPM, DDIM step scheduling, and CRPS computation. |
| [`dataset_loader.py`](dataset_loader.py) | **Data** | Sliding-window `RSRPDataset` and `DataLoader` with Min-Max normalization to $[-1, 1]$. |
| [`train.py`](train.py) | **Training** | Unified training pipeline supporting both DDPM and TimeGrad across 10 ms and 50 ms horizons. |
| [`train_diffusion.py`](train_diffusion.py) | **Training** | Minimal standalone training script for base `RSRPDiffusion`. |
| [`compare_all_models.py`](compare_all_models.py) | **Benchmark** | Master evaluation script comparing MAE, CRPS, FNR, latency, and speedups across models. |
| [`calculate_advanced_metric.py`](calculate_advanced_metric.py) | **Analysis** | Generates 7 publication figures: CRPS decomposition, CDFs, calibration curves, and temporal stability. |
| [`run_handover.py`](run_handover.py) | **Real-Time** | Live handover decision simulator executing the 3-tier probabilistic risk protocol on sample windows. |
| [`scan_for_handover.py`](scan_for_handover.py) | **Real-Time** | Sequential trace scanner triggering accelerated DDIM inference only when signal drops below $-100\text{ dBm}$. |
| [`test_overfitting.py`](test_overfitting.py) | **Validation** | Train MSE vs. Test MSE validation check across multiple drive-test measurement splits. |
| [`visualize_results.py`](visualize_results.py) | **Visualization** | Multi-panel plotting script displaying history, ground truth, sample clouds, and model overlays. |
| [`ddim_visualization.py`](ddim_visualization.py) | **Visualization** | Generates focused trajectory clouds for DDIM deterministic predictions. |
| [`diffusion_handover_model.pth`](diffusion_handover_model.pth) | **Model Weights** | Pretrained PyTorch model checkpoint (758 KB, 128 hidden dim). |
| [`thesis_rishi_v4.pdf`](thesis_rishi_v4.pdf) | **Documentation** | Full compiled M.Tech Thesis document submitted to IIIT Naya Raipur. |
| [`generate_rishi_presentation.py`](generate_rishi_presentation.py) | **Slides** | Programmatic LaTeX Beamer slide deck generator for thesis defense. |

---

## 🛠️ 8. Quickstart & Usage Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/rishi-th219/Proactive-Handover-Prediction-in-5G-using-Diffusion-Models.git
cd Proactive-Handover-Prediction-in-5G-using-Diffusion-Models

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install torch numpy pandas matplotlib
```

### 2. Dataset Setup
Create a `data/` directory and place the drive-test CSV files inside:
- `data/drive_test_measurements01.csv` (Training data)
- `data/drive_test_measurements03.csv` (Held-out test data)

*(Each CSV must contain an `RSRP` column with numeric dBm values).*

### 3. Model Training
To train DDPM or TimeGrad from scratch:
```bash
python train.py
```
*(Select `"ddpm"` or `"timegrad"` and `pred_len` (10 or 50) in `CONFIG` at the top of `train.py`).*

### 4. Running Model Comparisons
To benchmark all model variants on test data:
```bash
python compare_all_models.py
```
Outputs tabular summaries and comparative bar plots in `results/`.

### 5. Generate Thesis Publication Figures
To compute advanced metrics and produce the 7 paper figures:
```bash
python calculate_advanced_metric.py
```
Generates:
- `fig1_metrics_bar.png`: Key metric comparisons (MAE, CRPS, FNR, Speed)
- `fig2_error_distribution.png`: Per-window error distribution (Violin + Box plots)
- `fig3_cdf_error.png`: Cumulative Distribution Function of absolute prediction errors
- `fig4_calibration_curve.png`: Predicted risk probability vs actual failure rates
- `fig5_crps_breakdown.png`: CRPS decomposition (Accuracy vs Sharpness)
- `fig6_speed_vs_quality.png`: Speed vs accuracy scatter trade-off
- `fig7_temporal_stability.png`: Rolling temporal stability across evaluation windows

### 6. Test Real-Time Handover Execution
Simulate online risk-aware inference on an incoming signal window:
```bash
python run_handover.py
```

### 7. Visualize Trajectories
Plot side-by-side probabilistic trajectory clouds:
```bash
python visualize_results.py
```

---

## 📜 9. Citation

If you use this codebase, models, or methodology in your research, please cite:

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
This repository is licensed under the [MIT License](LICENSE) for research and academic use.
