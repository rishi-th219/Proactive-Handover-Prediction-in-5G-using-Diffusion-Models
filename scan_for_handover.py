import torch
import numpy as np
import pandas as pd
from diffusion_model import RSRPDiffusion
from dataset_loader import RSRPDataset, DataLoader

# --- CONFIGURATION ---
MODEL_PATH = "diffusion_handover_model.pth"
DATA_DIR = "data/"
TIMESTEPS = 100
THRESHOLD_DBM = -85 # Raised slightly to catch "Pre-failure" moments easier

# Load stats for conversion
df = pd.read_csv(DATA_DIR + 'drive_test_measurements01.csv')
RSRP_MIN = df['RSRP'].min()
RSRP_MAX = df['RSRP'].max()

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RSRPDiffusion().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

dataset = RSRPDataset(data_dir=DATA_DIR)
loader = DataLoader(dataset, batch_size=1, shuffle=False) # Sequential scan

# Setup Diffusion constants
betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

print(f"Scanning dataset for High Risk scenarios (Threshold: < {THRESHOLD_DBM} dBm)...")

# --- SCANNING LOOP ---
for i, (history, _) in enumerate(loader):
    history = history.to(device)
    
    # Quick check: If current signal is already super good (> -90), skip to save time
    current_dbm = history[0, -1, 0].item() * (RSRP_MAX - RSRP_MIN)/2 + (RSRP_MAX + RSRP_MIN)/2
    if current_dbm > -100:
        if i % 100 == 0: print(f"Step {i}: Signal Strong ({current_dbm:.1f} dBm) - Skipping detailed generation")
        continue
        
    print(f"\n⚠️ Step {i}: Signal Weak ({current_dbm:.1f} dBm) -> Running Diffusion Analysis...")
    
    with torch.no_grad():
        n_samples = 50
        history_expanded = history.repeat(n_samples, 1, 1)
        x = torch.randn(n_samples, 10, 1).to(device)
        
        for t in reversed(range(TIMESTEPS)):
            t_tensor = torch.full((n_samples,), t, device=device)
            pred_noise = model(x, t_tensor, history_expanded)
            
            alpha = alphas[t]
            alpha_bar = alphas_cumprod[t]
            beta = betas[t]
            noise = torch.randn_like(x) if t > 0 else 0
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_bar))) * pred_noise) + torch.sqrt(beta) * noise

    # Analyze
    future_dbm = ((x.cpu().numpy() + 1) / 2) * (RSRP_MAX - RSRP_MIN) + RSRP_MIN
    
    drops = 0
    for j in range(n_samples):
        if np.min(future_dbm[j]) < THRESHOLD_DBM:
            drops += 1
    
    prob = drops / n_samples
    
    if prob > 0.4:
        print(f"🚨 FOUND CRITICAL EVENT AT STEP {i}!")
        print(f"Probability of Failure: {prob*100:.1f}%")
        if prob > 0.8:
            print("🔴 ACTION: TRIGGER HANDOVER IMMEDIATELY")
        else:
            print("🟡 ACTION: PREPARE HANDOVER")
        break # Stop when we find one
    else:
        print(f"   Analysis: Safe (Risk: {prob*100:.1f}%)")

