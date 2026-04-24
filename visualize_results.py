import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diffusion_model import RSRPDiffusion
from dataset_loader import RSRPDataset, DataLoader

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RSRPDiffusion().to(device)
model.load_state_dict(torch.load("diffusion_handover_model.pth", map_location=device))
model.eval()

# Get Data
dataset = RSRPDataset(data_dir="data/")
loader = DataLoader(dataset, batch_size=1, shuffle=True) # Random sample
history, actual_future = next(iter(loader))
history = history.to(device)

# Generate 50 Futures
with torch.no_grad():
    n_samples = 50
    history_exp = history.repeat(n_samples, 1, 1)
    x = torch.randn(n_samples, 10, 1).to(device)
    betas = torch.linspace(1e-4, 0.02, 100).to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    for t in reversed(range(100)):
        t_tensor = torch.full((n_samples,), t, device=device)
        pred_noise = model(x, t_tensor, history_exp)
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]
        noise = torch.randn_like(x) if t > 0 else 0
        x = (1/torch.sqrt(alpha)) * (x - ((1-alpha)/torch.sqrt(1-alpha_bar)) * pred_noise) + torch.sqrt(beta)*noise

# PLOT
history_np = history.cpu().numpy().flatten()
future_np = actual_future.numpy().flatten()
generated_np = x.cpu().numpy().squeeze()

plt.figure(figsize=(10, 6))

# 1. Plot History
time_hist = np.arange(0, 50)
plt.plot(time_hist, history_np, label='Past History', color='black', linewidth=2)

# 2. Plot Generated Futures (The "Cloud")
time_fut = np.arange(50, 60)
for i in range(n_samples):
    plt.plot(time_fut, generated_np[i], color='blue', alpha=0.1) # Low alpha for "cloud" effect

# 3. Plot Actual Future (Truth)
plt.plot(time_fut, future_np, label='Actual Future', color='red', linewidth=2, linestyle='--')

plt.title("Diffusion Model Prediction: Generated Futures vs Reality")
plt.xlabel("Time Steps")
plt.ylabel("Normalized RSRP")
plt.legend()
plt.grid(True)
plt.savefig("diffusion_result_plot.png")
print("Plot saved to diffusion_result_plot.png")
