import os

filepath = r"d:\5g_timegrad\thesis_rishi_v3\rishi_presentation.tex"

presentation_content = r"""\documentclass[aspectratio=169]{beamer}
\usetheme{Madrid}
\usecolortheme{default}

% ── Packages ──────────────────────────────────────────────────
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tikz}

% ── Metadata ──────────────────────────────────────────────────
\title[DDIM for Proactive Handover]{High-Speed Generative Diffusion and DDIM Deterministic Sampling for Risk-Aware Handover in 5G Networks}
\subtitle{M.Tech Thesis Defense}
\author[Rishi Thakur]{Rishi Thakur \\ Roll No: 243010109 \\ \vspace{0.2cm} \small Supervisor: Dr. Srinivasa KG \\ Co-Supervisor: Dr. Mallikharjuna Rao K.}
\institute[IIIT Raipur]{Department of Data Science and Artificial Intelligence \\ IIIT Naya Raipur}
\date{June 2026}

\begin{document}

% Slide 1: Title
\frame{\titlepage}

% Slide 2: Outline
\begin{frame}{Presentation Outline}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Introduction \& Motivation
                \item Mobility Challenges in mmWave
                \item Reactive Trigger Bottlenecks
                \item Research Objectives
                \item Foundations of DDPM
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \begin{itemize}
                \item Accelerated Sampling via DDIM
                \item Mathematical Derivations
                \item Step-Skipping Scheduling
                \item Jetson Nano Edge Deployment
                \item Quantitative Results \& Discussion
            \end{itemize}
        \end{column}
    \end{columns}
\end{frame}

% Slide 3: Introduction
\begin{frame}{Introduction \& Network Evolution}
    \begin{itemize}
        \item \textbf{High-Frequency Bands:} 5G New Radio (NR) leverages millimeter-wave (mmWave) frequencies to meet massive bandwidth demands.
        \item \textbf{Dense Small Cells:} mmWave propagation limits cell coverage, necessitating dense deployments.
        \item \textbf{Mobility Management:} Frequent handovers occur as User Equipment (UE) moves between small cells.
        \item \textbf{Critical Goal:} Avoid link failures and session drops during mobility transitions.
    \end{itemize}
\end{frame}

% Slide 4: Mobility Challenges in mmWave
\begin{frame}{Mobility Challenges in mmWave Bands}
    \begin{itemize}
        \item \textbf{Propagation Characteristics:} High isotropic path loss and severe penetration losses.
        \item \textbf{Dynamic Obstacles:} Concrete structures, vehicles, and human blockages cause sudden signal drops.
        \item \textbf{Stochastic Fading:} Reference Signal Received Power (RSRP) drops by 20--30 dB in milliseconds.
        \item \textbf{Handover Window:} The time window to transition connectivity drops to under 50 ms.
    \end{itemize}
\end{frame}

% Slide 5: Limitations of Reactive Handover
\begin{frame}{Limitations of Reactive 3GPP Handover}
    \begin{itemize}
        \item \textbf{3GPP Event A3:} Reactive trigger comparing serving and neighbor RSRP margins.
        \item \textbf{Time-To-Trigger (TTT):} TTT margins (e.g., 80 ms to 512 ms) delay decisions too long during rapid shadowing.
        \item \textbf{Radio Link Failure (RLF):} UE drops connection before completing physical re-association.
        \item \textbf{Ping-Pong Effect:} Aggressive margins cause redundant switching, overloading the control plane.
    \end{itemize}
\end{frame}

% Slide 6: Proactive Handover Forecasting
\begin{frame}{Proactive Handover Forecasting}
    \begin{itemize}
        \item \textbf{Conceptual Shift:} Treat mobility management as a multi-variate signal forecasting problem.
        \item \textbf{Prediction Window:} Forecast future RSRP values before the link degrades below connection thresholds.
        \item \textbf{Pre-emptive Allocation:} SDN controllers prepare resources at the target base station in advance.
        \item \textbf{Zero-Loss Sessions:} Handover completes before the serving link experiences complete blockage.
    \end{itemize}
\end{frame}

% Slide 7: Research Objectives
\begin{frame}{Research Objectives}
    \begin{enumerate}
        \item \textbf{Generative Modeling:} Use temporal diffusion models to capture propagation uncertainty.
        \item \textbf{Latency Optimization:} Address the computational bottlenecks of diffusion models.
        \item \textbf{Deterministic Execution:} Establish a reproducible mapping from noise to predictions to stabilize control loop policies.
        \item \textbf{Edge Validation:} Deploy and profile the framework on a resource-constrained hardware processor (NVIDIA Jetson Nano).
    \end{enumerate}
\end{frame}

% Slide 8: Literature Review \& Baselines
\begin{frame}{Literature Review \& Baselines}
    \begin{itemize}
        \item \textbf{Statistical Models (ARIMA, Kalman Filters):} Fail to capture highly non-linear signal drops under blockage.
        \item \textbf{Recurrent Networks (LSTM, GRU):} Prone to error accumulation over long-horizon predictions.
        \item \textbf{Generative Adversarial Networks (GANs):} Fast inference but suffer from mode collapse and training instability.
        \item \textbf{Diffusion Models (DDPM):} State-of-the-art accuracy but suffer from extreme computational complexity.
    \end{itemize}
\end{frame}

% Slide 9: Foundations of DDPM
\begin{frame}{Foundations of Denoising Diffusion Probabilistic Models}
    \begin{itemize}
        \item \textbf{Forward Process:} Systematically adds Gaussian noise to the clean RSRP trajectory over $T$ steps:
        \begin{equation*}
            q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})
        \end{equation*}
        \item \textbf{Reverse Process:} A deep neural network learns to denoise the sequence:
        \begin{equation*}
            p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
        \end{equation*}
        \item \textbf{Uncertainty Handling:} Captured naturally by modeling the signal generation process probabilistically.
    \end{itemize}
\end{frame}

% Slide 10: Markovian Forward and Reverse Trajectories
\begin{frame}{Markovian Trajectories and Sequential Bottlenecks}
    \begin{itemize}
        \item \textbf{Markov Chain Assumption:} In DDPM, the forward noising process assumes each step depends strictly on the previous.
        \item \textbf{Reverse Step Dependency:} The reverse process must walk sequentially through all $T$ steps (typically $T=1000$).
        \item \textbf{Sequential Computations:} Evaluating the denoising network 1000 times accumulates latency.
        \item \textbf{Latency Impact:} Inference time takes several seconds, which is unusable for real-time 5G control loops.
    \end{itemize}
\end{frame}

% Slide 11: The Compute Latency Bottleneck
\begin{frame}{The Compute Latency Bottleneck}
    \begin{itemize}
        \item \textbf{5G Edge Constraints:} The total control loop latency budget is limited to under 50 ms.
        \item \textbf{Inference Delay:} Sequential DDPM evaluation exceeds the budget by 50x.
        \item \textbf{Problem:} How to maintain the generative accuracy of diffusion models while matching edge latency requirements?
        \item \textbf{Solution:} Break the Markovian assumption to accelerate sampling.
    \end{itemize}
\end{frame}

% Slide 12: Denoising Diffusion Implicit Models (DDIM)
\begin{frame}{Denoising Diffusion Implicit Models (DDIM)}
    \begin{itemize}
        \item \textbf{Non-Markovian Forward Process:} DDIM defines a family of forward processes that share the same marginal distributions as DDPM:
        \begin{equation*}
            q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t)\mathbf{I})
        \end{equation*}
        \item \textbf{Mathematical Leverage:} Models trained under the standard DDPM framework can be sampled using DDIM without retraining.
        \item \textbf{Key Feature:} Allows the reverse process to skip intermediate steps.
    \end{itemize}
\end{frame}

% Slide 13: Non-Markovian Forward Process Definition
\begin{frame}{Non-Markovian Forward Process Definition}
    \begin{itemize}
        \item The joint distribution of the forward process is defined as:
        \begin{equation*}
            q_\sigma(x_{1:T} \mid x_0) = q(x_T \mid x_0) \prod_{t=2}^T q_\sigma(x_{t-1} \mid x_t, x_0)
        \end{equation*}
        \item The conditional transitions are formulated to satisfy the marginals:
        \begin{equation*}
            q_\sigma(x_{t-1} \mid x_t, x_0) = \mathcal{N}\left( x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1 - \bar{\alpha}_t}}, \sigma_t^2 \mathbf{I} \right)
        \end{equation*}
        \item $\sigma_t^2$ is a control parameter regulating the stochasticity of the reverse trajectory.
    \end{itemize}
\end{frame}

% Slide 14: Mathematical Derivation of Deterministic Reverse Step
\begin{frame}{Mathematical Derivation of Deterministic Reverse Step}
    \begin{itemize}
        \item \textbf{Deterministic Sampling:} Setting $\sigma_t^2 = 0$ for all $t$ removes stochastic noise during generation.
        \item \textbf{Deterministic Update Rule:} The reverse step simplifies to:
        \begin{equation*}
            x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t, h_t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1-\bar{\alpha}_{t-1}} \epsilon_\theta(x_t, t, h_t)
        \end{equation*}
        \item \textbf{Consistency Benefit:} Maps a specific latent starting noise $x_T$ to a unique, stable generated trajectory, preventing policy oscillation.
    \end{itemize}
\end{frame}

% Slide 15: Skipping Steps via Sub-sequence Sampling
\begin{frame}{Step-Skipping and Sampling Acceleration}
    \begin{itemize}
        \item \textbf{Concept:} Evaluate the model only on a small subset of steps $\tau = [t_1, t_2, \dots, t_S]$, where $S \ll T$.
        \item \textbf{Reverse Transition:} The model transitions directly from $x_{t_i}$ to $x_{t_{i-1}}$, skipping all intermediate steps.
        \item \textbf{Complexity Reduction:} Neural network evaluations drop from $T = 1000$ to $S = 10$ or $S = 20$.
        \item \textbf{Inference Speedup:} Reduces computation time from seconds to milliseconds.
    \end{itemize}
\end{frame}

% Slide 16: Linear and Quadratic Step-skipping Schedules
\begin{frame}{Step-Skipping Scheduling Strategies}
    \begin{itemize}
        \item \textbf{Linear Schedule:} Steps are spaced uniformly:
        \begin{equation*}
            t_i = \lfloor i \times \frac{T}{S} \rfloor
        \end{equation*}
        \item \textbf{Quadratic Schedule:} Steps are clustered close to the clean data boundary ($t=0$):
        \begin{equation*}
            t_i = \lfloor T \times (i/S)^2 \rfloor
        \end{equation*}
        \item \textbf{Rationale:} Finer denoising updates at the end preserve trajectory shapes and transition patterns.
    \end{itemize}
\end{frame}

% Slide 17: GBDT-based RSRP Preprocessing and Dual-Masking
\begin{frame}{RSRP Preprocessing \& Dual-Masking}
    \begin{itemize}
        \item \textbf{RSRP Noise:} High-frequency fluctuations from multi-path fading interfere with trend learning.
        \item \textbf{Trend Extraction:} Gradient Boosted Decision Trees (GBDT) smooth raw RSRP measurements.
        \item \textbf{Dual-Masking:} Masks transient signal drops during loss evaluation, forcing the network to focus on long-term shadowing trends.
        \item \textbf{Result:} Improved stability in forecasted handover triggers.
    \end{itemize}
\end{frame}

% Slide 18: Objective Function: MAE vs Compute Latency
\begin{frame}{Objective Function: MAE vs Compute Latency}
    \begin{itemize}
        \item \textbf{Trade-off:} Reducing step size $S$ speeds up inference but introduces approximation errors.
        \item \textbf{Optimization Objective:} Minimize combined error and computational delay:
        \begin{equation*}
            \min_{S \in \mathcal{S}} \mathcal{L}(S) = \text{MAE}(S) + \lambda T_{\text{inference}}(S)
        \end{equation*}
        \item \textbf{Regularization ($\lambda$):} Regulates operator preferences for execution speed versus forecasting accuracy.
    \end{itemize}
\end{frame}

% Slide 19: Proposed SDN-RIC System Architecture
\begin{frame}{Proposed SDN-RIC System Architecture}
    \begin{itemize}
        \item \textbf{Telemetry Collection:} Base stations forward RRC reports containing serving and neighbor RSRP measurements.
        \item \textbf{Edge Inference Engine:} Runs TensorRT-optimized DDIM models to forecast signal levels.
        \item \textbf{Decision Module:} Evaluates predicted degradation and makes pre-emptive handover choices.
        \item \textbf{SDN Controller:} Ryu controller deploys OpenFlow flow table reconfigurations to redirect packets.
    \end{itemize}
\end{frame}

% Slide 20: Technical Implementation Stack
\begin{frame}{Technical Implementation \& Dataset}
    \begin{columns}
        \begin{column}{0.5\textwidth}
            \textbf{Software Stack:}
            \begin{itemize}
                \item PyTorch 2.1 \& TensorRT graph compilation.
                \item Ryu SDN Controller \& OpenFlow 1.3.
                \item gRPC API communication interfaces.
            \end{itemize}
        \end{column}
        \begin{column}{0.5\textwidth}
            \textbf{Dataset Details:}
            \begin{itemize}
                \item 300,000 RSRP samples from real drive tests and ns-3.
                \item Logs include: cell IDs, RSRQ, velocity, and blockage tags.
            \end{itemize}
        \end{column}
    \end{columns}
\end{frame}

% Slide 21: NVIDIA Jetson Nano Deployment and Profiling
\begin{frame}{Edge Hardware Deployment}
    \begin{itemize}
        \item \textbf{Platform:} NVIDIA Jetson Nano (128-core Maxwell GPU, 4-core ARM CPU).
        \item \textbf{Edge Mimicry:} Simulates the processing constraints of computational nodes at cell towers.
        \item \textbf{Optimization:} Graphs compiled into fp16 TensorRT engines to maximize edge throughput.
        \item \textbf{Metric profiled:} Direct inference execution delay ($T_{\text{inference}}$).
    \end{itemize}
\end{frame}

% Slide 22: ns-3 Simulation Layout and Scenarios
\begin{frame}{ns-3 Simulation Environment}
    \begin{itemize}
        \item \textbf{Layout:} 12 small-cell gNodeBs operating in the 28 GHz mmWave band.
        \item \textbf{Mobility:} High-speed rail and urban vehicle paths ranging from 30 km/h to 120 km/h.
        \item \textbf{Blockages:} Random building blockages causing severe shadow fading events.
        \item \textbf{Closed-loop Control:} Integrates the prediction engine directly with physical re-association.
    \end{itemize}
\end{frame}

% Slide 23: Evaluation Metrics
\begin{frame}{Evaluation Metrics}
    \begin{itemize}
        \item \textbf{Forecasting Accuracy:} Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
        \item \textbf{Compute Performance:} Model execution latency on edge hardware.
        \item \textbf{Networking KPIs:}
        \begin{itemize}
            \item \textbf{Handover Failure Rate:} Percentage of connection drops during cell transitions.
            \item \textbf{Ping-Pong Rate:} Rate of redundant re-associations.
            \item \textbf{Link Service Drop:} Service interruption duration.
        \end{itemize}
    \end{itemize}
\end{frame}

% Slide 24: Results: Latency-Accuracy Trade-off
\begin{frame}{Results: Latency-Accuracy Trade-off}
    \begin{table}[h]
    \centering
    \small
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Model} & \textbf{Steps (S)} & \textbf{MAE (dBm)} & \textbf{Latency (ms)} & \textbf{Speedup} \\
        \midrule
        Base DDPM & 1000 & 1.82 & 2650.0 & 1x \\
        DDIM-100 & 100 & 1.86 & 265.0 & 10x \\
        DDIM-50 & 50 & 1.91 & 132.5 & 20x \\
        DDIM-20 & 20 & 2.01 & 53.0 & 50x \\
        DDIM-10 & 10 & 2.12 & 26.5 & 100x \\
        \bottomrule
    \end{tabular}
    \end{table}
    \begin{itemize}
        \item DDIM-10 reduces latency to 26.5 ms, achieving a 100x speedup with minimal accuracy loss.
    \end{itemize}
\end{frame}

% Slide 25: Results: Trajectory Prediction Accuracy
\begin{frame}{Results: Trajectory Prediction Accuracy}
    \begin{table}[h]
    \centering
    \small
    \begin{tabular}{lccc}
        \toprule
        \textbf{Metric} & \textbf{DDIM-10 (Proposed)} & \textbf{LSTM Baseline} & \textbf{DLinear Baseline} \\
        \midrule
        MAE (dBm) & 2.12 & 3.45 & 4.12 \\
        RMSE (dBm) & 2.82 & 4.21 & 5.03 \\
        FDE (dBm) & 3.12 & 5.10 & 6.22 \\
        \bottomrule
    \end{tabular}
    \end{table}
    \begin{itemize}
        \item The proposed DDIM model reduces prediction error by 38\% compared to recurrent networks.
    \end{itemize}
\end{frame}

% Slide 26: Results: Networking KPIs
\begin{frame}{Results: Handover Performance comparison}
    \begin{table}[h]
    \centering
    \small
    \begin{tabular}{lccc}
        \toprule
        \textbf{Metric} & \textbf{Event A3 Reactive} & \textbf{LSTM Proactive} & \textbf{DDIM-10 Proactive} \\
        \midrule
        HO Failure Rate & 8.42\% & 3.12\% & 0.95\% \\
        Ping-Pong Rate & 12.10\% & 5.45\% & 1.22\% \\
        Link Service Drop & 5.22s & 1.84s & 0.12s \\
        \bottomrule
    \end{tabular}
    \end{table}
    \begin{itemize}
        \item Proactive DDIM-10 reduces handover failures by 75\% and limits ping-pong oscillations to 1.22\%.
    \end{itemize}
\end{frame}

% Slide 27: Results: Compute Performance on Jetson Nano
\begin{frame}{Results: Edge Resource Profiling}
    \begin{itemize}
        \item \textbf{GPU Core Utilization:} Averages 88\% on the Jetson Nano, demonstrating high parallel efficiency.
        \item \textbf{Memory Footprint:} Consumes 310 MB of GPU RAM, fitting within low-cost edge device limits.
        \item \textbf{Power Profile:} Averages 4.2 W under peak execution loads.
        \item \textbf{Thermal Profile:} Stays within safe bounds, making it suitable for uncooled tower enclosures.
    \end{itemize}
\end{frame}

% Slide 28: Results: SDN Control Loop Execution Time
\begin{frame}{Results: Control Loop Timing distribution}
    \begin{itemize}
        \item \textbf{Total Loop Latency:}
        \begin{equation*}
            T_{\text{loop}} = T_{\text{trans}} + T_{\text{inf}} + T_{\text{dec}} + T_{\text{policy}}
        \end{equation*}
        \item \textbf{Values:} $T_{\text{trans}} \approx 2.0$ ms, $T_{\text{dec}} \approx 0.5$ ms, $T_{\text{policy}} \approx 9.5$ ms.
        \item \textbf{DDIM-10 Inference:} $T_{\text{inf}} = 26.5$ ms $\rightarrow T_{\text{loop}} \approx 38.5$ ms.
        \item \textbf{Feasibility:} The 38.5 ms loop fits within the 50 ms budget, preventing link dropouts at 120 km/h.
    \end{itemize}
\end{frame}

% Slide 29: Future Directions
\begin{frame}{Future Directions \& O-RAN Integration}
    \begin{itemize}
        \item \textbf{O-RAN RIC Integration:} Deploying the model as an xApp in the Near-Real-Time RAN Intelligent Controller.
        \item \textbf{Consistency Models:} Distilling DDIM into single-step generation models to reduce inference to under 5 ms.
        \item \textbf{Continuous Neural ODEs:} Adaptively adjusting step size $S$ based on channel stability.
    \end{itemize}
\end{frame}

% Slide 30: Thank You
\begin{frame}{Thank You}
    \centering
    \Large Thank You! \\
    \vspace{0.8cm}
    \large Rishi Thakur \\
    Department of DSAI, IIIT Naya Raipur \\
    \vspace{0.5cm}
    \small Questions \& Discussion
\end{frame}

\end{document}
"""

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(presentation_content)

print("Rishi V3 slide presentation generated successfully.")
