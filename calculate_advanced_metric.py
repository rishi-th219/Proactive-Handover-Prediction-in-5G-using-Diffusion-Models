"""
calculate_advanced_metric.py
============================
Evaluates DDPM, DDIM-50, and TimeGrad on the test set and generates
7 publication-quality figures for the research paper.

Figures produced (all saved to results/):
  Fig 1 — fig1_metrics_bar.png          Main metrics bar chart (MAE, CRPS, FNR, Time)
  Fig 2 — fig2_error_distribution.png   Per-window error distribution (violin + box)
  Fig 3 — fig3_cdf_error.png            CDF of absolute prediction error
  Fig 4 — fig4_calibration_curve.png    Predicted risk probability vs actual failure rate
  Fig 5 — fig5_crps_breakdown.png       CRPS split into accuracy + sharpness terms
  Fig 6 — fig6_speed_vs_quality.png     Speed vs accuracy trade-off scatter
  Fig 7 — fig7_temporal_stability.png   Rolling MAE/CRPS over evaluation sequence
  CSV   — advanced_metrics_pred10.csv   Full numeric results table
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
TEST_FILE     = "data/drive_test_measurements03.csv"
THRESHOLD_DBM = -85
N_SAMPLES     = 50
N_EVAL        = 100
DDIM_STEPS    = 50
PRED_LEN      = 10

os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COLORS = {"DDPM": "#1f77b4", "DDIM-50": "#2ca02c", "TimeGrad": "#d62728"}

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
df_ref   = pd.read_csv(f"{DATA_DIR}drive_test_measurements01.csv")
RSRP_MIN = float(df_ref["RSRP"].min())
RSRP_MAX = float(df_ref["RSRP"].max())


def crps_components(forecasts, observation):
    """Returns (crps, mae_term, diversity_term)."""
    mae_term  = np.mean(np.abs(forecasts - observation))
    n         = len(forecasts)
    diversity = sum(
        np.mean(np.abs(forecasts[i] - forecasts[(i + 1) % n]))
        for i in range(n)
    ) / n
    return mae_term - 0.5 * diversity, mae_term, diversity


# ─────────────────────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────────────────────
print("Loading models...")

ddpm_model = RSRPDiffusion().to(device)
ddpm_model.load_state_dict(torch.load(DDPM_PATH, map_location=device))
ddpm_model.eval()
print(f"  DDPM     <- {DDPM_PATH}")

run_timegrad = os.path.exists(TIMEGRAD_PATH)
if run_timegrad:
    tg_model = TimeGradModel().to(device)
    tg_model.load_state_dict(torch.load(TIMEGRAD_PATH, map_location=device))
    tg_model.eval()
    print(f"  TimeGrad <- {TIMEGRAD_PATH}")
else:
    print(f"  [SKIP] TimeGrad not found: {TIMEGRAD_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation — collect per-window records
# ─────────────────────────────────────────────────────────────────────────────
dataset      = RSRPDataset(data_dir=DATA_DIR, pred_len=PRED_LEN)
df_test      = pd.read_csv(TEST_FILE)
dataset.data = df_test["RSRP"].values.astype(np.float32)
dataset.data = (dataset.data - RSRP_MIN) / (RSRP_MAX - RSRP_MIN) * 2 - 1
loader       = DataLoader(dataset, batch_size=1, shuffle=False)

betas, alphas, alphas_cumprod = build_noise_schedule(device)

records = {"DDPM": [], "DDIM-50": [], "TimeGrad": []}

print(f"\nEvaluating {N_EVAL} windows ...")

with torch.no_grad():
    count = 0
    for i, (history, actual_future) in enumerate(loader):
        if i % 10 != 0:
            continue
        if count >= N_EVAL:
            break

        history  = history.float().to(device)
        real_np  = actual_future.squeeze().numpy()
        real_dbm = denorm(real_np, RSRP_MIN, RSRP_MAX)

        runs = [
            ("DDPM",    lambda: ddpm_sample(ddpm_model, history, N_SAMPLES, PRED_LEN,
                                             betas, alphas, alphas_cumprod, device)),
            ("DDIM-50", lambda: ddim_sample(ddpm_model, history, N_SAMPLES, PRED_LEN,
                                             alphas_cumprod, ddim_steps=DDIM_STEPS,
                                             eta=0.0, device=device)),
        ]
        if run_timegrad:
            runs.append(
                ("TimeGrad", lambda: timegrad_sample(tg_model, history, N_SAMPLES, PRED_LEN,
                                                      betas, alphas, alphas_cumprod, device))
            )

        for name, sample_fn in runs:
            x, elapsed = sample_fn()
            gen_dbm    = denorm(x.cpu().numpy().squeeze(-1), RSRP_MIN, RSRP_MAX)
            mean_pred  = np.mean(gen_dbm, axis=0)
            mae        = float(np.mean(np.abs(real_dbm - mean_pred)))
            crps, mae_term, diversity = crps_components(gen_dbm, real_dbm)
            real_fail  = bool(np.min(real_dbm) < THRESHOLD_DBM)
            risk_prob  = float(np.sum(np.min(gen_dbm, axis=1) < THRESHOLD_DBM) / N_SAMPLES)
            model_fail = risk_prob > 0.4

            records[name].append({
                "mae"       : mae,
                "crps"      : crps,
                "mae_term"  : mae_term,
                "diversity" : diversity,
                "elapsed"   : elapsed,
                "risk_prob" : risk_prob,
                "real_fail" : real_fail,
                "missed"    : real_fail and not model_fail,
                "window_idx": count,
            })

        count += 1
        print(f"  Window {count}/{N_EVAL}", end="\r")

print(f"\nDone.\n")

active = [n for n in ["DDPM", "DDIM-50", "TimeGrad"] if records[n]]

def a(name, key):
    return np.array([r[key] for r in records[name]])

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
summary_rows = []
ddpm_time    = None

for name in active:
    n          = len(records[name])
    true_fails = int(a(name, "real_fail").sum())
    missed     = int(a(name, "missed").sum())
    fnr        = missed / true_fails * 100 if true_fails > 0 else 0.0
    avg_time   = float(a(name, "elapsed").mean())
    if name == "DDPM":
        ddpm_time = avg_time
    summary_rows.append({
        "Model"        : name,
        "MAE (dB)"     : round(float(a(name, "mae").mean()),  4),
        "CRPS"         : round(float(a(name, "crps").mean()), 4),
        "FNR (%)"      : round(fnr, 2),
        "Time/win (s)" : round(avg_time, 4),
        "True Failures": true_fails,
        "Missed"       : missed,
        "N"            : n,
        "_time"        : avg_time,
    })

for r in summary_rows:
    r["Speedup"] = round(ddpm_time / r["_time"], 2) if ddpm_time else 1.0

df_sum = pd.DataFrame(summary_rows)
SHOW   = ["Model", "MAE (dB)", "CRPS", "FNR (%)", "Time/win (s)", "Speedup"]
print("=" * 65)
print("RESULTS  — pred_len=10 (10 ms horizon)")
print("=" * 65)
print(df_sum[SHOW].to_string(index=False))
print("=" * 65)
df_sum[SHOW].to_csv("results/advanced_metrics_pred10.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — Main metrics bar chart
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle("Model Comparison — Key Metrics (pred_len=10, 10 ms horizon)",
             fontsize=12, fontweight="bold")

panels = [
    ("mae",   "MAE (dB)",            "lower is better"),
    ("crps",  "CRPS Score",          "lower is better"),
    (None,    "False Negative Rate (%)", "lower is better"),
    ("elapsed","Inference Time (s)", "lower is better"),
]

for ax, (key, title, note) in zip(axes, panels):
    if key is None:
        vals  = [r["FNR (%)"]   for r in summary_rows]
        names = [r["Model"]     for r in summary_rows]
    else:
        names = active
        vals  = [float(a(n, key).mean()) for n in names]
    cols  = [COLORS[n] for n in names]
    bars  = ax.bar(names, vals, color=cols, edgecolor="black",
                   linewidth=0.8, alpha=0.85, width=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(note, fontsize=8, color="grey")
    ax.tick_params(axis="x", rotation=20, labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.4g}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("results/fig1_metrics_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> results/fig1_metrics_bar.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — Per-window error distribution (violin + box)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Per-Window Error Distribution", fontsize=12, fontweight="bold")

for ax, (key, label) in zip(axes, [("mae", "MAE per window (dB)"),
                                    ("crps","CRPS per window")]):
    data   = [a(n, key) for n in active]
    colors = [COLORS[n] for n in active]

    vp = ax.violinplot(data, positions=range(len(active)),
                       showmedians=False, showextrema=False)
    for body, col in zip(vp["bodies"], colors):
        body.set_facecolor(col);  body.set_alpha(0.35)

    bp = ax.boxplot(data, positions=range(len(active)),
                    widths=0.22, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col);  patch.set_alpha(0.75)

    ax.set_xticks(range(len(active)))
    ax.set_xticklabels(active, fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label,  fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for j, d in enumerate(data):
        med = float(np.median(d))
        ax.text(j + 0.14, med, f"{med:.3f}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig("results/fig2_error_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> results/fig2_error_distribution.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — CDF of absolute prediction error
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("Cumulative Distribution of Prediction Error",
             fontsize=11, fontweight="bold")

for name in active:
    errors = np.sort(a(name, "mae"))
    cdf    = np.arange(1, len(errors) + 1) / len(errors)
    ax.plot(errors, cdf, color=COLORS[name], linewidth=2.5, label=name)
    for pct, ls in [(50, "--"), (90, ":")]:
        v = float(np.percentile(errors, pct))
        ax.axvline(v, color=COLORS[name], linestyle=ls, linewidth=0.9, alpha=0.55)

ax.plot([], [], "k--", linewidth=1, label="50th percentile")
ax.plot([], [], "k:",  linewidth=1, label="90th percentile")
ax.set_xlabel("MAE (dB)", fontsize=11)
ax.set_ylabel("Cumulative Probability", fontsize=11)
ax.legend(fontsize=9);  ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig3_cdf_error.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> results/fig3_cdf_error.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — Calibration curve
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 6))
ax.set_title("Risk Calibration Curve\n"
             "(Predicted Failure Probability vs Actual Failure Rate)",
             fontsize=11, fontweight="bold")

bins = np.linspace(0, 1, 6)
for name in active:
    probs  = a(name, "risk_prob")
    actual = a(name, "real_fail").astype(float)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() >= 3:
            xs.append((lo + hi) / 2)
            ys.append(float(actual[mask].mean()))
    if xs:
        ax.plot(xs, ys, "o-", color=COLORS[name], linewidth=2,
                markersize=8, label=name)

ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
ax.set_xlabel("Predicted Failure Probability", fontsize=11)
ax.set_ylabel("Actual Failure Rate",           fontsize=11)
ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
ax.legend(fontsize=10);  ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fig4_calibration_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> results/fig4_calibration_curve.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — CRPS decomposition
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("CRPS Decomposition: Accuracy Term vs Sharpness/Diversity Term",
             fontsize=11, fontweight="bold")

x_pos = np.arange(len(active));  w = 0.28
acc_v = [float(a(n, "mae_term").mean())  for n in active]
div_v = [float(a(n, "diversity").mean()) for n in active]
net_v = [float(a(n, "crps").mean())      for n in active]

b1 = ax.bar(x_pos - w/2, acc_v, w, label="Accuracy term",
            color=[COLORS[n] for n in active], alpha=0.9,
            edgecolor="black", linewidth=0.8)
b2 = ax.bar(x_pos + w/2, div_v, w, label="Diversity (sharpness)",
            color=[COLORS[n] for n in active], alpha=0.4,
            edgecolor="black", linewidth=0.8, hatch="//")
for j, val in enumerate(net_v):
    ax.plot(x_pos[j], val, "D", color="black", markersize=9, zorder=5,
            label="Net CRPS" if j == 0 else "")

ax.set_xticks(x_pos);  ax.set_xticklabels(active, fontsize=10)
ax.set_ylabel("Score (lower is better)", fontsize=10)
ax.legend(fontsize=9);  ax.grid(axis="y", alpha=0.3)
for bar, val in zip(list(b1) + list(b2), acc_v + div_v):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("results/fig5_crps_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> results/fig5_crps_breakdown.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — Speed vs quality scatter
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Inference Speed vs Prediction Quality Trade-off",
             fontsize=12, fontweight="bold")

for ax, (key, ylabel) in zip(axes, [("mae",  "MAE (dB)"),
                                     ("crps", "CRPS Score")]):
    for name in active:
        t = float(a(name, "elapsed").mean())
        q = float(a(name, key).mean())
        ax.scatter(t, q, s=280, color=COLORS[name], zorder=5,
                   edgecolors="black", linewidth=1.2, label=name)
        ax.annotate(name, (t, q), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Avg Inference Time / Window (s)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"Time vs {ylabel}", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/fig6_speed_vs_quality.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> results/fig6_speed_vs_quality.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 — Temporal stability (rolling MAE + CRPS)
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
fig.suptitle("Temporal Stability over Evaluation Sequence",
             fontsize=12, fontweight="bold")

W = 10
for name in active:
    for ax, key, label in [(axes[0], "mae", "Rolling MAE (dB)"),
                            (axes[1], "crps", "Rolling CRPS")]:
        seq  = a(name, key)
        roll = np.convolve(seq, np.ones(W) / W, mode="valid")
        ax.plot(np.arange(len(roll)), roll,
                color=COLORS[name], linewidth=2, label=name)

for ax, label in [(axes[0], "Rolling MAE (dB)"),
                   (axes[1], "Rolling CRPS")]:
    # Shade windows where real signal failed
    for name in active:
        for r in records[name]:
            if r["real_fail"] and r["window_idx"] >= W - 1:
                ax.axvspan(r["window_idx"] - W + 1, r["window_idx"] - W + 2,
                           alpha=0.08, color="red")
        break   # shade once (same windows for all models)
    ax.set_ylabel(label, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

axes[0].set_title(f"Rolling MAE  (window = {W})", fontsize=10)
axes[1].set_title(f"Rolling CRPS (window = {W})", fontsize=10)
axes[1].set_xlabel("Evaluation Window Index", fontsize=10)
fig.text(0.5, 0.01,
         "Red shaded regions = windows where real RSRP crossed failure threshold",
         ha="center", fontsize=8, color="grey")
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("results/fig7_temporal_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> results/fig7_temporal_stability.png")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("ALL OUTPUTS SAVED TO results/")
print("=" * 55)
for fname, desc in [
    ("fig1_metrics_bar.png",        "Main metrics bar chart"),
    ("fig2_error_distribution.png", "Violin + box error distribution"),
    ("fig3_cdf_error.png",          "CDF of prediction error"),
    ("fig4_calibration_curve.png",  "Risk calibration curve"),
    ("fig5_crps_breakdown.png",     "CRPS decomposition"),
    ("fig6_speed_vs_quality.png",   "Speed vs quality scatter"),
    ("fig7_temporal_stability.png", "Rolling MAE/CRPS stability"),
    ("advanced_metrics_pred10.csv", "Numeric results table"),
]:
    print(f"  {fname:<38} {desc}")