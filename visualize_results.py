"""
visualize_results.py
====================
Plots history + actual future against prediction clouds from
DDPM, DDIM-50, and TimeGrad side by side.

Layout: 1 row, 3 panels (one per model), all on the same RSRP scale.
Also generates a 4th overlay panel showing all three models together.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from diffusion_model import RSRPDiffusion
from timegrad_model  import TimeGradModel
from dataset_loader  import RSRPDataset, DataLoader
from ddim_sampler    import (build_noise_schedule, ddpm_sample,
                              ddim_sample, timegrad_sample, denorm)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DDPM_PATH     = "diffusion_handover_model.pth"
TIMEGRAD_PATH = "models/timegrad_pred10_model.pth"
DATA_DIR      = "data/"
N_SAMPLES     = 50
PRED_LEN      = 10
DDIM_STEPS    = 50
OUTPUT_FILE   = "results/model_comparison_plot.png"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Normalisation stats ───────────────────────────────────────────────────────
df_ref   = pd.read_csv(f"{DATA_DIR}drive_test_measurements01.csv")
RSRP_MIN = float(df_ref["RSRP"].min())
RSRP_MAX = float(df_ref["RSRP"].max())

# ── Load models ───────────────────────────────────────────────────────────────
ddpm_model = RSRPDiffusion().to(device)
ddpm_model.load_state_dict(torch.load(DDPM_PATH, map_location=device))
ddpm_model.eval()

run_timegrad = os.path.exists(TIMEGRAD_PATH)
if run_timegrad:
    tg_model = TimeGradModel().to(device)
    tg_model.load_state_dict(torch.load(TIMEGRAD_PATH, map_location=device))
    tg_model.eval()
else:
    print(f"[INFO] TimeGrad checkpoint not found ({TIMEGRAD_PATH}). "
          "Plotting DDPM and DDIM only.")

# ── Data ─────────────────────────────────────────────────────────────────────
dataset = RSRPDataset(data_dir=DATA_DIR, pred_len=PRED_LEN)
loader  = DataLoader(dataset, batch_size=1, shuffle=True)
history_norm, actual_future_norm = next(iter(loader))
history_norm = history_norm.to(device)

betas, alphas, alphas_cumprod = build_noise_schedule(device)

# ── Generate futures ─────────────────────────────────────────────────────────
print("Generating DDPM futures ...")
with torch.no_grad():
    x_ddpm, t_ddpm = ddpm_sample(ddpm_model, history_norm, N_SAMPLES, PRED_LEN,
                                  betas, alphas, alphas_cumprod, device)

print("Generating DDIM-50 futures ...")
with torch.no_grad():
    x_ddim, t_ddim = ddim_sample(ddpm_model, history_norm, N_SAMPLES, PRED_LEN,
                                  alphas_cumprod, ddim_steps=DDIM_STEPS,
                                  eta=0.0, device=device)

if run_timegrad:
    print("Generating TimeGrad futures ...")
    with torch.no_grad():
        x_tg, t_tg = timegrad_sample(tg_model, history_norm, N_SAMPLES, PRED_LEN,
                                      betas, alphas, alphas_cumprod, device)

# ── Denormalise ───────────────────────────────────────────────────────────────
def to_dbm(x_tensor):
    return denorm(x_tensor.cpu().numpy().squeeze(-1), RSRP_MIN, RSRP_MAX)

hist_dbm   = denorm(history_norm.cpu().numpy().squeeze(), RSRP_MIN, RSRP_MAX)
future_dbm = denorm(actual_future_norm.numpy().squeeze(), RSRP_MIN, RSRP_MAX)

ddpm_dbm = to_dbm(x_ddpm)      # (N_SAMPLES, PRED_LEN)
ddim_dbm = to_dbm(x_ddim)
tg_dbm   = to_dbm(x_tg) if run_timegrad else None

# ── Time axes ────────────────────────────────────────────────────────────────
seq_len   = len(hist_dbm)
t_hist    = np.arange(seq_len)
t_fut     = np.arange(seq_len, seq_len + PRED_LEN)

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
n_panels = 4 if run_timegrad else 3   # individual + overlay
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
fig.suptitle("RSRP Probabilistic Forecast: DDPM vs DDIM-50 vs TimeGrad",
             fontsize=12, fontweight="bold")


def draw_panel(ax, gen_dbm, model_label, cloud_color, inference_time=None):
    # History
    ax.plot(t_hist, hist_dbm, color="black", linewidth=2, label="History", zorder=4)
    # Ground truth
    ax.plot(t_fut, future_dbm, color="red", linewidth=2,
            linestyle="--", label="Actual Future", zorder=5)
    # Prediction cloud
    for s in range(len(gen_dbm)):
        ax.plot(t_fut, gen_dbm[s], color=cloud_color, alpha=0.12, linewidth=0.8)
    # Mean prediction
    ax.plot(t_fut, np.mean(gen_dbm, axis=0), color=cloud_color,
            linewidth=2.5, linestyle="-", label="Mean Prediction", zorder=6)
    # Threshold line
    ax.axhline(y=-85, color="orange", linestyle=":", linewidth=1.2,
               label="Threshold (−85 dBm)")
    # Divider
    ax.axvline(x=seq_len - 0.5, color="grey", linestyle="--", linewidth=1.0)

    title = model_label
    if inference_time is not None:
        title += f"\n({inference_time:.3f}s/window)"
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Time Step (ms)", fontsize=9)
    ax.set_ylabel("RSRP (dBm)", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)


draw_panel(axes[0], ddpm_dbm, "DDPM (100 steps)",  "#1f77b4", t_ddpm)
draw_panel(axes[1], ddim_dbm, f"DDIM-{DDIM_STEPS} steps", "#2ca02c", t_ddim)
if run_timegrad:
    draw_panel(axes[2], tg_dbm, "TimeGrad (autoregressive)", "#d62728", t_tg)

# Overlay panel — all three models' mean on one axis
ax_ov = axes[-1]
ax_ov.plot(t_hist, hist_dbm, color="black", linewidth=2, label="History", zorder=4)
ax_ov.plot(t_fut, future_dbm, color="red", linewidth=2,
           linestyle="--", label="Actual Future", zorder=5)
ax_ov.plot(t_fut, np.mean(ddpm_dbm, axis=0), color="#1f77b4",
           linewidth=2, label=f"DDPM mean ({t_ddpm:.3f}s)")
ax_ov.plot(t_fut, np.mean(ddim_dbm, axis=0), color="#2ca02c",
           linewidth=2, label=f"DDIM-50 mean ({t_ddim:.3f}s)")
if run_timegrad:
    ax_ov.plot(t_fut, np.mean(tg_dbm, axis=0), color="#d62728",
               linewidth=2, label=f"TimeGrad mean ({t_tg:.3f}s)")
ax_ov.axhline(y=-85, color="orange", linestyle=":", linewidth=1.2,
              label="Threshold")
ax_ov.axvline(x=seq_len - 0.5, color="grey", linestyle="--", linewidth=1.0)
ax_ov.set_title("Mean Predictions Overlay", fontsize=10, fontweight="bold")
ax_ov.set_xlabel("Time Step (ms)", fontsize=9)
ax_ov.legend(fontsize=7, loc="lower left")
ax_ov.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {OUTPUT_FILE}")

# ── Print timing summary ──────────────────────────────────────────────────────
print("\nInference time per window:")
print(f"  DDPM      : {t_ddpm:.4f}s  (baseline)")
print(f"  DDIM-50   : {t_ddim:.4f}s  ({t_ddpm/t_ddim:.1f}× faster)")
if run_timegrad:
    print(f"  TimeGrad  : {t_tg:.4f}s  ({t_ddpm/t_tg:.2f}× vs DDPM)")
