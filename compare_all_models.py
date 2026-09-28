"""
compare_all_models.py
=====================
Master comparison script. Evaluates all model variants and produces
the full results table for your paper.

Group 1 — pred_len=10 (10 ms horizon)
    DDPM-10      baseline parallel diffusion   (diffusion_handover_model.pth)
    DDIM-10      same weights, 50-step sampler  (diffusion_handover_model.pth)
    TimeGrad-10  autoregressive diffusion       (models/timegrad_pred10_model.pth)

Group 2 — pred_len=50 (50 ms horizon)
    DDPM-50      parallel diffusion             (models/ddpm_pred50_model.pth)
    TimeGrad-50  autoregressive diffusion       (models/timegrad_pred50_model.pth)

Metrics: MAE (dB), CRPS, FNR (%), Inference Time (s/window), Speedup vs DDPM
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
CHECKPOINTS = {
    "DDPM-10"      : "diffusion_handover_model.pth",
    "DDIM-10"      : "diffusion_handover_model.pth",   # same weights as DDPM-10
    "TimeGrad-10"  : "models/timegrad_pred10_model.pth",
    "DDPM-50"      : "models/ddpm_pred50_model.pth",
    "TimeGrad-50"  : "models/timegrad_pred50_model.pth",
}

DATA_DIR      = "data/"
TEST_FILE     = "data/drive_test_measurements03.csv"
THRESHOLD_DBM = -85
N_SAMPLES     = 50
N_EVAL        = 100     # evaluation windows per model
DDIM_STEPS    = 50

os.makedirs("results", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_norm_stats():
    df = pd.read_csv(f"{DATA_DIR}drive_test_measurements01.csv")
    return float(df["RSRP"].min()), float(df["RSRP"].max())

RSRP_MIN, RSRP_MAX = get_norm_stats()


def crps(forecasts, observation):
    """
    forecasts   : (N_SAMPLES, pred_len)
    observation : (pred_len,)
    """
    mae_term  = np.mean(np.abs(forecasts - observation))
    n         = len(forecasts)
    diversity = sum(
        np.mean(np.abs(forecasts[i] - forecasts[(i + 1) % n]))
        for i in range(n)
    ) / n
    return mae_term - 0.5 * diversity


def load_model(variant, ckpt_path):
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] {variant}: checkpoint not found → {ckpt_path}")
        return None
    cls = TimeGradModel if "TimeGrad" in variant else RSRPDiffusion
    m   = cls()
    m.load_state_dict(torch.load(ckpt_path, map_location=device))
    m.to(device).eval()
    print(f"  Loaded {variant:15s} <- {ckpt_path}")
    return m


def build_test_loader(pred_len):
    dataset      = RSRPDataset(data_dir=DATA_DIR, pred_len=pred_len)
    df_test      = pd.read_csv(TEST_FILE)
    dataset.data = df_test["RSRP"].values.astype(np.float32)
    dataset.data = (dataset.data - RSRP_MIN) / (RSRP_MAX - RSRP_MIN) * 2 - 1
    return DataLoader(dataset, batch_size=1, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(variant, model, pred_len):
    betas, alphas, alphas_cumprod = build_noise_schedule(device)
    loader = build_test_loader(pred_len)

    total_mae = total_crps = total_time = 0.0
    true_fail = missed = count = 0

    with torch.no_grad():
        for i, (history, actual_future) in enumerate(loader):
            if i % 10 != 0:
                continue
            if count >= N_EVAL:
                break

            history = history.float().to(device)               # (1, seq_len, 1)
            real_np  = actual_future.squeeze().numpy()         # (pred_len,)
            real_dbm = denorm(real_np, RSRP_MIN, RSRP_MAX)

            # ── Sample ───────────────────────────────────────────────────────
            if variant == "DDIM-10":
                x, elapsed = ddim_sample(
                    model, history, N_SAMPLES, pred_len,
                    alphas_cumprod, ddim_steps=DDIM_STEPS, eta=0.0, device=device
                )
            elif "TimeGrad" in variant:
                x, elapsed = timegrad_sample(
                    model, history, N_SAMPLES, pred_len,
                    betas, alphas, alphas_cumprod, device
                )
            else:   # DDPM-10 or DDPM-50
                x, elapsed = ddpm_sample(
                    model, history, N_SAMPLES, pred_len,
                    betas, alphas, alphas_cumprod, device
                )

            total_time += elapsed

            gen_np  = x.cpu().numpy().squeeze(-1)              # (N_SAMPLES, pred_len)
            gen_dbm = denorm(gen_np, RSRP_MIN, RSRP_MAX)

            # ── Metrics ──────────────────────────────────────────────────────
            mean_pred  = np.mean(gen_dbm, axis=0)
            total_mae  += np.mean(np.abs(real_dbm - mean_pred))
            total_crps += crps(gen_dbm, real_dbm)

            real_fail_flag = np.min(real_dbm) < THRESHOLD_DBM
            risk           = np.sum(np.min(gen_dbm, axis=1) < THRESHOLD_DBM) / N_SAMPLES
            model_fail     = risk > 0.4

            if real_fail_flag:
                true_fail += 1
                if not model_fail:
                    missed += 1

            count += 1

        if count == 0:
            return None

    fnr      = (missed / true_fail * 100) if true_fail > 0 else 0.0
    avg_time = total_time / count

    return {
        "Model"        : variant,
        "pred_len"     : pred_len,
        "MAE (dB)"     : round(total_mae  / count, 4),
        "CRPS"         : round(total_crps / count, 4),
        "FNR (%)"      : round(fnr, 2),
        "Time/win (s)" : round(avg_time, 4),
        "True Fails"   : true_fail,
        "Missed"       : missed,
        "_time_raw"    : avg_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run all evaluations
# ─────────────────────────────────────────────────────────────────────────────
PRED_LEN = {"DDPM-10": 10, "DDIM-10": 10, "TimeGrad-10": 10,
            "DDPM-50": 50, "TimeGrad-50": 50}

print("Loading models...")
models = {v: load_model(v, ck) for v, ck in CHECKPOINTS.items()}

print("\nRunning evaluations (this takes a few minutes)...")
results = {}
for variant, model in models.items():
    if model is None:
        continue
    print(f"  Evaluating {variant}...", end=" ", flush=True)
    r = evaluate(variant, model, PRED_LEN[variant])
    if r:
        results[variant] = r
        print(f"MAE={r['MAE (dB)']:.4f}  CRPS={r['CRPS']:.4f}  "
              f"FNR={r['FNR (%)']:.1f}%  Time={r['Time/win (s)']:.4f}s")

if not results:
    print("No results — train models first using train.py")
    exit()

# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────
def add_speedup(group):
    rows = [r for r in results.values() if r["pred_len"] == group]
    if not rows:
        return rows
    baseline = next((r for r in rows if "DDPM" in r["Model"] and "TimeGrad" not in r["Model"]), rows[0])
    for r in rows:
        sp = baseline["_time_raw"] / r["_time_raw"] if r["_time_raw"] > 0 else 1.0
        r["Speedup"] = round(sp, 2)
    return rows

g1 = add_speedup(10)
g2 = add_speedup(50)

DISPLAY_COLS = ["Model", "MAE (dB)", "CRPS", "FNR (%)", "Time/win (s)", "Speedup"]

print("\n" + "=" * 70)
print("GROUP 1 — pred_len=10 (10 ms horizon)")
print("=" * 70)
if g1:
    df1 = pd.DataFrame(g1)[DISPLAY_COLS]
    print(df1.to_string(index=False))
    df1.to_csv("results/group1_pred10_results.csv", index=False)

print("\n" + "=" * 70)
print("GROUP 2 — pred_len=50 (50 ms horizon)")
print("=" * 70)
if g2:
    df2 = pd.DataFrame(g2)[DISPLAY_COLS]
    print(df2.to_string(index=False))
    df2.to_csv("results/group2_pred50_results.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "DDPM-10"     : "#1f77b4",
    "DDIM-10"     : "#2ca02c",
    "TimeGrad-10" : "#d62728",
    "DDPM-50"     : "#1f77b4",
    "TimeGrad-50" : "#d62728",
}


def bar_group(ax, data_rows, metric, title, ylabel, highlight_low=True):
    names  = [r["Model"]  for r in data_rows]
    values = [r[metric]   for r in data_rows]
    colors = [COLORS.get(n, "#7f7f7f") for n in names]
    bars   = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4g}", ha="center", va="bottom", fontsize=8)


def plot_group(group_rows, group_label, filename):
    if not group_rows:
        return
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    fig.suptitle(f"Model Comparison — {group_label}", fontsize=12, fontweight="bold")
    bar_group(axes[0], group_rows, "MAE (dB)",     "MAE (dB)",       "dB")
    bar_group(axes[1], group_rows, "CRPS",         "CRPS",           "Score (lower=better)")
    bar_group(axes[2], group_rows, "FNR (%)",      "False Negative Rate", "% (lower=better)")
    bar_group(axes[3], group_rows, "Time/win (s)", "Inference Time", "seconds (lower=better)")
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved results/{filename}")


def plot_speedup(group_rows, group_label, filename):
    if not group_rows or "Speedup" not in group_rows[0]:
        return
    names   = [r["Model"]   for r in group_rows]
    speedup = [r["Speedup"] for r in group_rows]
    colors  = [COLORS.get(n, "#7f7f7f") for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, speedup, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="DDPM baseline")
    ax.set_title(f"Speedup vs DDPM — {group_label}", fontsize=10, fontweight="bold")
    ax.set_ylabel("Speedup factor (×)", fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, speedup):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}×", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"results/{filename}", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved results/{filename}")


print("\nGenerating plots...")
plot_group(g1, "pred_len=10 (10 ms)", "group1_comparison.png")
plot_group(g2, "pred_len=50 (50 ms)", "group2_comparison.png")
plot_speedup(g1, "pred_len=10",        "group1_speedup.png")
plot_speedup(g2, "pred_len=50",        "group2_speedup.png")

# ── Combined summary plot ─────────────────────────────────────────────────────
all_rows = list(results.values())
if all_rows:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Full Comparison — All Models & Horizons", fontsize=12, fontweight="bold")

    def _bar(ax, rows, metric, title):
        bar_group(ax, rows, metric, title, metric)
        ax.set_xlabel("")

    _bar(axes[0], all_rows, "MAE (dB)",     "MAE (dB) — All Models")
    _bar(axes[1], all_rows, "CRPS",         "CRPS — All Models")
    _bar(axes[2], all_rows, "Time/win (s)", "Inference Time — All Models")

    # Shade background by group
    for ax in axes:
        ax.axvline(2.5, color="grey", linestyle=":", linewidth=1.0)
        ax.text(1.0, ax.get_ylim()[1] * 0.97, "pred_len=10", ha="center",
                fontsize=7, color="grey")
        ax.text(3.5, ax.get_ylim()[1] * 0.97, "pred_len=50", ha="center",
                fontsize=7, color="grey")

    plt.tight_layout()
    plt.savefig("results/full_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved results/full_comparison.png")

print("\n✓ All results saved to results/")
print("  CSV files ready to copy into your paper tables.")
