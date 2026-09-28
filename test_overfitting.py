"""
test_overfitting.py
===================
Compares train-set MSE vs test-set MSE for all model variants.
A large gap = overfitting. Similar values = good generalisation.

Models tested (if checkpoint exists):
    DDPM-10, DDIM-10, TimeGrad-10   (pred_len=10)
    DDPM-50, TimeGrad-50            (pred_len=50)
"""

import torch
import numpy as np
import pandas as pd
import os

from diffusion_model import RSRPDiffusion
from timegrad_model  import TimeGradModel
from dataset_loader  import RSRPDataset, DataLoader
from ddim_sampler    import (build_noise_schedule, ddpm_sample,
                              ddim_sample, timegrad_sample)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FILE = "data/drive_test_measurements01.csv"
TEST_FILE  = "data/drive_test_measurements03.csv"
DATA_DIR   = "data/"
N_SAMPLES  = 10      # fewer samples for speed; increase for accuracy
N_BATCHES  = 10      # batches to evaluate per split
DDIM_STEPS = 50

VARIANTS = [
    # (label,          checkpoint_path,                      model_cls,     pred_len, sampler)
    ("DDPM-10",     "diffusion_handover_model.pth",          "ddpm",        10,       "ddpm"),
    ("DDIM-10",     "diffusion_handover_model.pth",          "ddpm",        10,       "ddim"),
    ("TimeGrad-10", "models/timegrad_pred10_model.pth",      "timegrad",    10,       "timegrad"),
    ("DDPM-50",     "models/ddpm_pred50_model.pth",          "ddpm",        50,       "ddpm"),
    ("TimeGrad-50", "models/timegrad_pred50_model.pth",      "timegrad",    50,       "timegrad"),
]
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(file_path, pred_len):
    """Build a DataLoader from a single CSV file (re-normalised per file)."""
    dataset = RSRPDataset(data_dir=DATA_DIR, pred_len=pred_len)
    df      = pd.read_csv(file_path)
    arr     = df["RSRP"].values.astype(np.float32)
    mn, mx  = arr.min(), arr.max()
    dataset.data = (arr - mn) / (mx - mn) * 2 - 1
    return DataLoader(dataset, batch_size=32, shuffle=False)


def evaluate_mse(model, loader, sampler_type, pred_len,
                 betas, alphas, alphas_cumprod):
    """Return mean MSE between mean-prediction and actual future."""
    total_mse = 0.0
    count     = 0

    with torch.no_grad():
        for history, actual_future in loader:
            history       = history.to(device)
            actual_future = actual_future.to(device)
            B             = history.shape[0]

            # Expand for n_samples predictions per batch item
            # (stack samples, then average → mean prediction per item)
            results = []
            for b in range(B):
                h_b = history[b:b+1]    # (1, seq_len, 1)

                if sampler_type == "ddim":
                    x, _ = ddim_sample(model, h_b, N_SAMPLES, pred_len,
                                        alphas_cumprod, ddim_steps=DDIM_STEPS,
                                        eta=0.0, device=device)
                elif sampler_type == "timegrad":
                    x, _ = timegrad_sample(model, h_b, N_SAMPLES, pred_len,
                                            betas, alphas, alphas_cumprod, device)
                else:
                    x, _ = ddpm_sample(model, h_b, N_SAMPLES, pred_len,
                                        betas, alphas, alphas_cumprod, device)

                mean_pred = x.mean(dim=0)          # (pred_len, 1)
                results.append(mean_pred.unsqueeze(0))

            mean_preds = torch.cat(results, dim=0) # (B, pred_len, 1)
            mse        = torch.mean((mean_preds - actual_future) ** 2).item()
            total_mse += mse
            count     += 1
            if count >= N_BATCHES:
                break

    return total_mse / count if count > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("OVERFITTING TEST - Train MSE vs Test MSE")
print("=" * 60)
print(f"{'Model':<15}  {'Train MSE':>10}  {'Test MSE':>10}  {'Gap':>10}  {'Status':>12}")
print("-" * 60)

results = []

for label, ckpt, model_cls, pred_len, sampler in VARIANTS:
    if not os.path.exists(ckpt):
        print(f"{label:<15}  {'N/A':>10}  {'N/A':>10}  {'-':>10}  checkpoint missing")
        continue

    # Load model
    m = (RSRPDiffusion() if model_cls == "ddpm" else TimeGradModel()).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()

    betas, alphas, alphas_cumprod = build_noise_schedule(device)

    train_loader = make_loader(TRAIN_FILE, pred_len)
    test_loader  = make_loader(TEST_FILE,  pred_len)

    train_mse = evaluate_mse(m, train_loader, sampler, pred_len,
                              betas, alphas, alphas_cumprod)
    test_mse  = evaluate_mse(m, test_loader,  sampler, pred_len,
                              betas, alphas, alphas_cumprod)

    gap    = test_mse - train_mse
    ratio  = test_mse / train_mse if train_mse > 0 else float("nan")
    status = "OK" if ratio < 2.0 else "OVERFIT"

    print(f"{label:<15}  {train_mse:>10.5f}  {test_mse:>10.5f}  "
          f"{gap:>+10.5f}  {status:>12}")

    results.append({
        "Model"     : label,
        "pred_len"  : pred_len,
        "Train MSE" : round(train_mse, 5),
        "Test MSE"  : round(test_mse,  5),
        "Gap"       : round(gap,       5),
        "Ratio"     : round(ratio,     3),
        "Status"    : status,
    })

print("=" * 60)
print("Gap = Test MSE - Train MSE  |  Ratio = Test/Train  |  Threshold: <2.0")

if results:
    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/overfitting_test.csv", index=False)
    print("\nSaved → results/overfitting_test.csv")
