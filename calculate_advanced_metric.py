import torch
import numpy as np
import pandas as pd
from diffusion_model import RSRPDiffusion
from dataset_loader import RSRPDataset, DataLoader

# --- CONFIG ---
MODEL_PATH = "diffusion_handover_model.pth"
# Use the Test Set (Unseen Data) for fair comparison
DATA_FILE = "data/drive_test_measurements03.csv" 
THRESHOLD_DBM = -85 # The "Failure" line
N_SAMPLES = 50       # Number of futures to generate

# --- HELPERS ---
def calculate_crps(forecasts, observation):
    """
    forecasts: shape (N_SAMPLES, PREDICTION_LEN)
    observation: shape (PREDICTION_LEN)
    Returns single scalar CRPS score for this window.
    """
    # 1. Mean Absolute Error (MAE) term
    # Average distance between each sample and the truth
    mae_term = np.mean(np.abs(forecasts - observation))
    
    # 2. Diversity term (Spread)
    # Average distance between samples themselves
    # Efficient calculation using sorting not strictly needed for small N, 
    # but let's use a simplified pair-wise estimate for clarity
    diversity_term = 0
    count = 0
    # To save time, we just sample a few pairs instead of all 50x50
    for i in range(len(forecasts)):
        # Compare sample i against the next sample (cyclic)
        next_i = (i + 1) % len(forecasts)
        diversity_term += np.mean(np.abs(forecasts[i] - forecasts[next_i]))
        count += 1
    
    diversity_term = diversity_term / count
    
    return mae_term - (0.5 * diversity_term)

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Stats for Denormalization (Approximate from training)
# Ideally you loaded these from a config file
df_temp = pd.read_csv("data/drive_test_measurements01.csv")
RSRP_MIN = df_temp['RSRP'].min()
RSRP_MAX = df_temp['RSRP'].max()

# Load Data
dataset = RSRPDataset(data_dir="data/")
# Override to point to specific test file
df_test = pd.read_csv(DATA_FILE)
dataset.data = df_test['RSRP'].values.astype(np.float32)
dataset.data = (dataset.data - RSRP_MIN) / (RSRP_MAX - RSRP_MIN) * 2 - 1
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load Model
model = RSRPDiffusion().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- METRIC ACCUMULATORS ---
total_crps = 0
total_mae = 0
count = 0

# Confusion Matrix for False Negatives
# Condition: Real Signal < THRESHOLD_DBM
true_failures = 0     # Real signal actually failed
predicted_failures = 0 # Model predicted failure (Prob > 50%)
missed_failures = 0   # False Negatives (Real Failed, Model said Safe)

print(f"Scanning {DATA_FILE} for Metrics...")
print(f"Threshold for Failure: {THRESHOLD_DBM} dBm")

# --- DIFFUSION CONSTANTS ---
betas = torch.linspace(1e-4, 0.02, 100).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

with torch.no_grad():
    for i, (history, actual_future) in enumerate(loader):
        # Optimization: Only test every 10th sample to save compute time
        # (Generating 50 futures for every millisecond takes forever)
        if i % 10 != 0: continue
        if count >= 100: break # Stop after 100 evaluation points for speed
        
        history = history.float().to(device)
        actual_future_np = actual_future.squeeze().numpy()
        
        # Denormalize Truth
        real_future_dbm = ((actual_future_np + 1) / 2) * (RSRP_MAX - RSRP_MIN) + RSRP_MIN
        
        # 1. GENERATE
        n_samples = N_SAMPLES
        history_exp = history.repeat(n_samples, 1, 1)
        x = torch.randn(n_samples, 10, 1).float().to(device)
        
        for t in reversed(range(100)):
            t_tensor = torch.full((n_samples,), t, device=device)
            pred_noise = model(x, t_tensor, history_exp)
            alpha = alphas[t]
            alpha_bar = alphas_cumprod[t]
            beta = betas[t]
            noise = torch.randn_like(x) if t > 0 else 0
            x = (1/torch.sqrt(alpha)) * (x - ((1-alpha)/torch.sqrt(1-alpha_bar)) * pred_noise) + torch.sqrt(beta)*noise
            
        # Denormalize Predictions
        gen_futures_np = x.cpu().numpy().squeeze()
        gen_futures_dbm = ((gen_futures_np + 1) / 2) * (RSRP_MAX - RSRP_MIN) + RSRP_MIN
        
        # 2. CALCULATE MAE (Accuracy)
        mean_prediction = np.mean(gen_futures_dbm, axis=0)
        mae = np.mean(np.abs(real_future_dbm - mean_prediction))
        total_mae += mae
        
        # 3. CALCULATE CRPS (Probabilistic Score)
        crps = calculate_crps(gen_futures_dbm, real_future_dbm)
        total_crps += crps
        
        # 4. CHECK FALSE NEGATIVES
        # Did the REAL signal actually drop below threshold at any point in the 10ms window?
        real_failure = np.min(real_future_dbm) < THRESHOLD_DBM
        
        # Did the MODEL predict high probability of failure?
        # Let's say: If >40% of scenarios fail, we call it a "Predicted Failure"
        failures_in_scenarios = np.sum(np.min(gen_futures_dbm, axis=1) < THRESHOLD_DBM)
        risk_prob = failures_in_scenarios / N_SAMPLES
        model_failure_pred = risk_prob > 0.4
        
        if real_failure:
            true_failures += 1
            if not model_failure_pred:
                missed_failures += 1 # OUCH! False Negative
                print(f"⚠️ False Negative at step {i}: Risk {risk_prob*100:.1f}% but Real dropped to {np.min(real_future_dbm):.1f}")
                
        count += 1
        print(f"Eval Step {count}: MAE={mae:.2f} | CRPS={crps:.2f}", end='\r')

print("\n" + "="*30)
print("FINAL RESULTS")
print("="*30)
print(f"Data Points Evaluated: {count}")
print(f"Mean Absolute Error (MAE): {total_mae/count:.4f} dB")
print(f"Mean CRPS Score:         {total_crps/count:.4f} (Lower is better)")
print("-" * 30)
print(f"Total True Failures (Real Data): {true_failures}")
print(f"Missed Failures (False Negatives): {missed_failures}")
if true_failures > 0:
    fnr = missed_failures / true_failures
    print(f"False Negative Rate (FNR): {fnr*100:.2f}%")
else:
    print("False Negative Rate (FNR): N/A (No failures in this test slice)")