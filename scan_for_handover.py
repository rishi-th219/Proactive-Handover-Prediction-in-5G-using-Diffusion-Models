import time

import numpy as np
import pandas as pd
import torch

from dataset_loader import DataLoader, RSRPDataset
from diffusion_model import RSRPDiffusion
from ddim import build_schedule, denormalize_rsrp, sample_futures

# ---------------- CONFIG ----------------
MODEL_PATH = "diffusion_handover_model.pth"
DATA_DIR = "data/"
TIMESTEPS = 100
BETA_START = 1e-4
BETA_END = 0.02
THRESHOLD_DBM = -85
FUTURE_LEN = 10
N_SAMPLES = 50
DDIM_STEPS = 50
DDIM_ETA = 0.0
SAMPLER = "ddim"  # "ddpm" or "ddim"

# ---------------- SETUP ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RSRPDiffusion().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

train_dataset = RSRPDataset(data_dir=DATA_DIR)
rsrp_min = float(train_dataset.min_val)
rsrp_max = float(train_dataset.max_val)

loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
schedule = build_schedule(TIMESTEPS, BETA_START, BETA_END, device=device)

print(f"Scanning dataset for high-risk scenarios (threshold: < {THRESHOLD_DBM} dBm)...")
print(f"Sampler: {SAMPLER.upper()}")

for i, (history, _) in enumerate(loader):
    history = history.to(device)
    current_dbm = denormalize_rsrp(history[0, -1, 0].item(), rsrp_min, rsrp_max)

    if current_dbm > -100:
        if i % 100 == 0:
            print(f"Step {i}: signal strong ({current_dbm:.1f} dBm) - skipping detailed generation")
        continue

    print(f"\n⚠️ Step {i}: signal weak ({current_dbm:.1f} dBm) -> running diffusion analysis...")

    with torch.no_grad():
        generator = torch.Generator(device=device)
        generator.manual_seed(999 + i)
        start = time.perf_counter()
        futures_norm = sample_futures(
            model=model,
            history=history,
            future_len=FUTURE_LEN,
            sampler=SAMPLER,
            schedule=schedule,
            n_samples=N_SAMPLES,
            num_inference_steps=DDIM_STEPS,
            eta=DDIM_ETA,
            device=device,
            generator=generator,
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

    future_dbm = denormalize_rsrp(futures_norm.cpu().numpy().squeeze(), rsrp_min, rsrp_max)
    prob = float(np.mean(np.min(future_dbm, axis=1) < THRESHOLD_DBM))

    print(f"   Risk: {prob * 100:.1f}% | latency: {latency_ms:.2f} ms")

    if prob > 0.4:
        print(f"🚨 FOUND CRITICAL EVENT AT STEP {i}!")
        if prob > 0.8:
            print("🔴 ACTION: TRIGGER HANDOVER IMMEDIATELY")
        else:
            print("🟡 ACTION: PREPARE HANDOVER")
        break
    else:
        print("   Analysis: safe")
