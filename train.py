"""
train.py — Unified training script for all model variants.
===========================================================

Edit the CONFIG block below, then run:
    python train.py

Four variants to train (run once each with different CONFIG):
──────────────────────────────────────────────────────────────
  model_type │ pred_len │  Checkpoint saved to
  ───────────┼──────────┼──────────────────────────────────────
  "ddpm"     │   10     │  (already done → diffusion_handover_model.pth)
  "timegrad" │   10     │  models/timegrad_pred10_model.pth
  "ddpm"     │   50     │  models/ddpm_pred50_model.pth
  "timegrad" │   50     │  models/timegrad_pred50_model.pth

NOTE: The existing diffusion_handover_model.pth IS your DDPM pred_len=10.
      No need to retrain it — skip that variant.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim

from dataset_loader import RSRPDataset, DataLoader
from diffusion_model import RSRPDiffusion
from timegrad_model  import TimeGradModel

# ─────────────────────────────────────────────────────────────────────────────
# ▼▼▼  EDIT THIS BLOCK TO SELECT WHAT TO TRAIN  ▼▼▼
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "model_type" : "timegrad",  # "ddpm" | "timegrad"
    "pred_len"   : 50,          # 10 | 50

    # ── Hyperparameters (same for all variants) ──
    "seq_len"    : 50,
    "epochs"     : 50,
    "batch_size" : 32,
    "lr"         : 1e-3,
    "timesteps"  : 100,
    "beta_start" : 1e-4,
    "beta_end"   : 0.02,
    "data_dir"   : "data/",
}
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
SAVE_PATH = f"models/{CONFIG['model_type']}_pred{CONFIG['pred_len']}_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 55)
print(f"  Model    : {CONFIG['model_type'].upper()}")
print(f"  pred_len : {CONFIG['pred_len']} steps  ({CONFIG['pred_len']} ms)")
print(f"  Epochs   : {CONFIG['epochs']}")
print(f"  Device   : {device}")
print(f"  Saving   : {SAVE_PATH}")
print("=" * 55)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = RSRPDataset(
    data_dir = CONFIG["data_dir"],
    seq_len  = CONFIG["seq_len"],
    pred_len = CONFIG["pred_len"],
)
loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)
print(f"Dataset  : {len(dataset):,} samples  |  {len(loader)} batches/epoch\n")

# ── Model ─────────────────────────────────────────────────────────────────────
if CONFIG["model_type"] == "ddpm":
    model = RSRPDiffusion().to(device)
elif CONFIG["model_type"] == "timegrad":
    model = TimeGradModel().to(device)
else:
    raise ValueError(f"Unknown model_type '{CONFIG['model_type']}'. Use 'ddpm' or 'timegrad'.")

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}\n")

optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
loss_fn   = nn.MSELoss()

# ── Noise schedule ────────────────────────────────────────────────────────────
T      = CONFIG["timesteps"]
betas  = torch.linspace(CONFIG["beta_start"], CONFIG["beta_end"], T, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


def add_noise(x_0, t):
    """Forward diffusion: corrupt x_0 at timestep t, return (x_noisy, noise)."""
    noise = torch.randn_like(x_0)
    ab    = alphas_cumprod[t].view(-1, 1, 1)
    x_t   = ab.sqrt() * x_0 + (1.0 - ab).sqrt() * noise
    return x_t, noise


# ── Training loop ─────────────────────────────────────────────────────────────
# NOTE: The training loop is IDENTICAL for DDPM and TimeGrad.
# Both are called as model(noisy_future, t, history).
# The architectural difference (parallel vs autoregressive context)
# is handled internally in each model's forward().

print("Training...")
best_loss  = float("inf")
best_epoch = 0

for epoch in range(1, CONFIG["epochs"] + 1):
    model.train()
    total_loss = 0.0

    for history, future in loader:
        history, future = history.to(device), future.to(device)

        t = torch.randint(0, T, (history.shape[0],), device=device)
        noisy_future, noise = add_noise(future, t)

        pred_noise = model(noisy_future, t, history)   # same API for both models
        loss       = loss_fn(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    saved    = ""

    if avg_loss < best_loss:
        best_loss  = avg_loss
        best_epoch = epoch
        torch.save(model.state_dict(), SAVE_PATH)
        saved = "  ← best"

    print(f"Epoch {epoch:3d}/{CONFIG['epochs']}  |  Loss: {avg_loss:.5f}{saved}")

print(f"\nDone. Best loss = {best_loss:.5f}  (epoch {best_epoch})")
print(f"Checkpoint saved → {SAVE_PATH}")
