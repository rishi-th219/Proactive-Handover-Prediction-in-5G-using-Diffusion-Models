import matplotlib.pyplot as plt
import numpy as np
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
FUTURE_LEN = 10
N_SAMPLES = 50
DDIM_STEPS = 50
DDIM_ETA = 0.0

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL ----------------
model = RSRPDiffusion().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---------------- DATA ----------------
dataset = RSRPDataset(data_dir=DATA_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=True)
history, actual_future = next(iter(loader))
history = history.to(device)

rsrp_min = float(dataset.min_val)
rsrp_max = float(dataset.max_val)
schedule = build_schedule(TIMESTEPS, BETA_START, BETA_END, device=device)

generator = torch.Generator(device=device)
generator.manual_seed(1234)

with torch.no_grad():
    samples = sample_futures(
        model=model,
        history=history,
        future_len=FUTURE_LEN,
        sampler="ddim",
        schedule=schedule,
        n_samples=N_SAMPLES,
        num_inference_steps=DDIM_STEPS,
        eta=DDIM_ETA,
        device=device,
        generator=generator,
    )

history_np = denormalize_rsrp(history.cpu().numpy().flatten(), rsrp_min, rsrp_max)
future_np = denormalize_rsrp(actual_future.numpy().flatten(), rsrp_min, rsrp_max)
samples_np = denormalize_rsrp(samples.cpu().numpy().squeeze(), rsrp_min, rsrp_max)

plt.figure(figsize=(10, 6))

time_hist = np.arange(0, len(history_np))
time_fut = np.arange(len(history_np), len(history_np) + len(future_np))

plt.plot(time_hist, history_np, label="Past History", color="black", linewidth=2)
for i in range(N_SAMPLES):
    plt.plot(time_fut, samples_np[i], color="blue", alpha=0.10)
plt.plot(time_fut, future_np, label="Actual Future", color="red", linewidth=2, linestyle="--")

plt.title(f"DDIM Generated Futures ({DDIM_STEPS} steps)")
plt.xlabel("Time Steps")
plt.ylabel("RSRP (dBm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ddim_result_plot.png", dpi=200)
print("Plot saved to ddim_result_plot.png")
