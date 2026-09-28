"""
run_handover.py
===============
Real-time handover decision system.
Generates N probabilistic futures and outputs a risk-based decision.

Change MODEL_TYPE below to switch between models:
    "ddpm"      — standard parallel diffusion (100 steps)
    "ddim"      — fast parallel diffusion (50 steps, same DDPM weights)
    "timegrad"  — autoregressive diffusion (requires separate training)
"""

import torch
import numpy as np
import pandas as pd

from diffusion_model import RSRPDiffusion
from timegrad_model  import TimeGradModel
from dataset_loader  import RSRPDataset, DataLoader
from ddim_sampler    import (build_noise_schedule, ddpm_sample,
                              ddim_sample, timegrad_sample, denorm)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — change MODEL_TYPE to switch models
# ─────────────────────────────────────────────────────────────────────────────
MODEL_TYPE    = "ddpm"      # "ddpm" | "ddim" | "timegrad"
PRED_LEN      = 10
THRESHOLD_DBM = -110
N_SAMPLES     = 50
DATA_DIR      = "data/"
DDIM_STEPS    = 50

CHECKPOINTS = {
    "ddpm"     : "diffusion_handover_model.pth",
    "ddim"     : "diffusion_handover_model.pth",   # same weights as ddpm
    "timegrad" : "models/timegrad_pred10_model.pth",
}
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Normalisation stats ───────────────────────────────────────────────────────
df_ref   = pd.read_csv(f"{DATA_DIR}drive_test_measurements01.csv")
RSRP_MIN = float(df_ref["RSRP"].min())
RSRP_MAX = float(df_ref["RSRP"].max())

# ── Load model ────────────────────────────────────────────────────────────────
ckpt_path = CHECKPOINTS[MODEL_TYPE]
if MODEL_TYPE == "timegrad":
    model = TimeGradModel().to(device)
else:
    model = RSRPDiffusion().to(device)

model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Model     : {MODEL_TYPE.upper()}")
print(f"Checkpoint: {ckpt_path}")

# ── Load test sequence ────────────────────────────────────────────────────────
dataset  = RSRPDataset(data_dir=DATA_DIR, pred_len=PRED_LEN)
loader   = DataLoader(dataset, batch_size=1, shuffle=False)

# Skip 500 samples to reach a more interesting signal region
iterator = iter(loader)
for _ in range(500):
    history, _ = next(iterator)
history = history.to(device)

# Current signal in dBm
current_norm = history[0, -1, 0].item()
current_dbm  = denorm(current_norm, RSRP_MIN, RSRP_MAX)
print(f"\nCurrent RSRP: {current_dbm:.2f} dBm")
print(f"Generating {N_SAMPLES} probabilistic futures ({PRED_LEN} ms ahead)...")

# ── Generate futures ──────────────────────────────────────────────────────────
betas, alphas, alphas_cumprod = build_noise_schedule(device)

with torch.no_grad():
    if MODEL_TYPE == "ddim":
        x, elapsed = ddim_sample(model, history, N_SAMPLES, PRED_LEN,
                                  alphas_cumprod, ddim_steps=DDIM_STEPS,
                                  eta=0.0, device=device)
    elif MODEL_TYPE == "timegrad":
        x, elapsed = timegrad_sample(model, history, N_SAMPLES, PRED_LEN,
                                      betas, alphas, alphas_cumprod, device)
    else:   # ddpm
        x, elapsed = ddpm_sample(model, history, N_SAMPLES, PRED_LEN,
                                  betas, alphas, alphas_cumprod, device)

future_dbm = denorm(x.cpu().numpy().squeeze(-1), RSRP_MIN, RSRP_MAX)  # (N_SAMPLES, PRED_LEN)

# ── Risk assessment ───────────────────────────────────────────────────────────
drops         = np.sum(np.min(future_dbm, axis=1) < THRESHOLD_DBM)
prob_failure  = drops / N_SAMPLES
min_predicted = np.min(future_dbm)
mean_future   = np.mean(future_dbm)

print(f"\nInference time : {elapsed:.4f}s")
print(f"Min predicted  : {min_predicted:.2f} dBm")
print(f"Mean predicted : {mean_future:.2f} dBm")
print(f"Risk (P < {THRESHOLD_DBM} dBm): {prob_failure * 100:.1f}%")

print("\n" + "─" * 35)
print("HANDOVER DECISION")
print("─" * 35)
if prob_failure > 0.8:
    print("🔴  TRIGGER HANDOVER IMMEDIATELY")
    print("    Reason: >80% of futures predict signal failure.")
elif prob_failure > 0.4:
    print("🟡  PREPARE HANDOVER (Measurement Gap)")
    print("    Reason: Signal unstable — moderate failure risk.")
else:
    print("🟢  STAY CONNECTED")
    print("    Reason: Signal predicted to remain stable.")
print("─" * 35)
