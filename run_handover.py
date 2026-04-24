import torch
import numpy as np
import pandas as pd
from diffusion_model import RSRPDiffusion
from dataset_loader import RSRPDataset, DataLoader

# --- CONFIGURATION ---
MODEL_PATH = "diffusion_handover_model.pth"
DATA_DIR = "data/"  # Folder where your CSVs are
TIMESTEPS = 100
THRESHOLD_DBM = -110  # Real world threshold for handover
# Note: We need the min/max from training to convert back to dBm.
# Based on typical RSRP values, let's estimate or recalculate them.
# Ideally, save these during training, but recalculating here works for a demo.
df = pd.read_csv(DATA_DIR + 'drive_test_measurements01.csv')
RSRP_MIN = df['RSRP'].min()
RSRP_MAX = df['RSRP'].max()

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = RSRPDiffusion().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model Loaded Successfully.")

# Load Data (We just grab one sequence to test)
dataset = RSRPDataset(data_dir=DATA_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Get the first batch of history (Real usage would be live data)
# Skip first 500 samples to find a bad signal area
iterator = iter(loader)
for _ in range(500): 
    history, _ = next(iterator)
history = history.to(device)

# --- GENERATION (Phase 3) ---
print(f"Simulating future for current signal: {history[0, -1, 0].item() * (RSRP_MAX - RSRP_MIN)/2 + (RSRP_MAX + RSRP_MIN)/2:.2f} dBm")
print("Generating 50 probabilistic future scenarios...")

# Define diffusion schedule (Must match training!)
betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

with torch.no_grad():
    n_samples = 50
    # Replicate the single history 50 times
    history_expanded = history.repeat(n_samples, 1, 1)
    
    # Start with pure noise
    x = torch.randn(n_samples, 10, 1).to(device) # 10 steps into the future
    
    # Reverse Diffusion (Denoising)
    for t in reversed(range(TIMESTEPS)):
        t_tensor = torch.full((n_samples,), t, device=device)
        pred_noise = model(x, t_tensor, history_expanded)
        
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        beta = betas[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
            
        x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * pred_noise) + torch.sqrt(beta) * noise

# --- DECISION (Phase 4) ---
# Convert normalized [-1, 1] output back to real dBm values
# Formula: x_norm = (x_real - min) / (max - min) * 2 - 1
# Reverse: x_real = ((x_norm + 1) / 2) * (max - min) + min
future_dbm = ((x.cpu().numpy() + 1) / 2) * (RSRP_MAX - RSRP_MIN) + RSRP_MIN

# Calculate Risk
drops = 0
risk_threshold = THRESHOLD_DBM 

for i in range(n_samples):
    # Check if ANY point in the future path drops below threshold
    if np.min(future_dbm[i]) < risk_threshold:
        drops += 1

prob_failure = drops / n_samples
print("-" * 30)
print(f"Analysis of 50 generated futures:")
print(f"Probability of dropping below {risk_threshold} dBm: {prob_failure * 100:.1f}%")

print("\n--- FINAL DECISION ---")
if prob_failure > 0.8:
    print("🔴 ACTION: TRIGGER HANDOVER IMMEDIATELY")
    print("Reason: High certainty of signal failure.")
elif prob_failure > 0.4:
    print("🟡 ACTION: PREPARE HANDOVER (Measurement Gap)")
    print("Reason: Signal is unstable, monitor closely.")
else:
    print("🟢 ACTION: STAY CONNECTED")
    print("Reason: Signal is predicted to remain stable.")
print("-" * 30)
