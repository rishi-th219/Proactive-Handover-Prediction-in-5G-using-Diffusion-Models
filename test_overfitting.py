import torch
import numpy as np
import pandas as pd
from diffusion_model import RSRPDiffusion
from dataset_loader import RSRPDataset, DataLoader

# CONFIG
MODEL_PATH = "diffusion_handover_model.pth"
TRAIN_FILE = "data/drive_test_measurements01.csv" # Seen during training
TEST_FILE = "data/drive_test_measurements03.csv"  # NEVER seen during training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_dataset(model, file_path, desc):
    dataset = RSRPDataset(data_dir="data/") 
    # HACK: Manually override data to point to just ONE file for testing
    df = pd.read_csv(file_path)
    # Assume column is RSRP based on previous fixes
    dataset.data = df['RSRP'].values.astype(np.float32)
    # Re-normalize using the same logic as training (Crucial!)
    # In a real pipeline, save these stats. Here we approximate for the test.
    min_val, max_val = dataset.data.min(), dataset.data.max()
    dataset.data = (dataset.data - min_val) / (max_val - min_val) * 2 - 1
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    total_mse = 0
    count = 0
    
    # Simple Metric: MSE of the "Mean Prediction" vs "Actual"
    with torch.no_grad():
        for history, actual_future in loader:
            history = history.to(device)
            actual_future = actual_future.to(device)
            
            # Fast Generation (10 samples per point to save time)
            n_samples = 10
            history_exp = history.repeat_interleave(n_samples, dim=0)
            
            # Generate... (Simplified Loop)
            x = torch.randn(history_exp.shape[0], 10, 1).to(device)
            betas = torch.linspace(1e-4, 0.02, 100).to(device)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            
            for t in reversed(range(100)):
                t_tensor = torch.full((x.shape[0],), t, device=device)
                pred_noise = model(x, t_tensor, history_exp)
                alpha = alphas[t]
                alpha_bar = alphas_cumprod[t]
                beta = betas[t]
                noise = torch.randn_like(x) if t > 0 else 0
                x = (1/torch.sqrt(alpha)) * (x - ((1-alpha)/torch.sqrt(1-alpha_bar)) * pred_noise) + torch.sqrt(beta)*noise
            
            # Average the 10 samples to get the "Expected" path
            x = x.view(history.shape[0], n_samples, 10, 1)
            mean_pred = x.mean(dim=1)
            
            # Calculate Error
            mse = torch.mean((mean_pred - actual_future)**2).item()
            total_mse += mse
            count += 1
            if count > 10: break # Only test first 10 batches
            
    print(f"[{desc}] MSE Error: {total_mse/count:.5f}")

# Load Model
model = RSRPDiffusion().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print("--- OVERFITTING TEST ---")
evaluate_dataset(model, TRAIN_FILE, "Training Data (Seen)")
evaluate_dataset(model, TEST_FILE, "Test Data (Unseen)")
