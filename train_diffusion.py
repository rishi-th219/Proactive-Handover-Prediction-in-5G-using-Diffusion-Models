import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import RSRPDataset, DataLoader
from diffusion_model import RSRPDiffusion

TIMESTEPS = 100
BETA_START = 1e-4
BETA_END = 0.02

# Load dataset and model FIRST
dataset = RSRPDataset(data_dir='data/')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RSRPDiffusion().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# NOW create noise schedule on the correct device
betas = torch.linspace(BETA_START, BETA_END, TIMESTEPS).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)

def add_noise(x_0, t):
    noise = torch.randn_like(x_0)
    sqrt_alpha_bar = torch.sqrt(alphas_cumprod[t])
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alphas_cumprod[t])
    sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1)
    sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1)
    return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

print("Starting Training...")
for epoch in range(50):
    total_loss = 0
    for history, future in loader:
        history, future = history.to(device), future.to(device)
        t = torch.randint(0, TIMESTEPS, (history.shape[0],)).to(device)
        noisy_future, noise = add_noise(future, t)
        pred_noise = model(noisy_future, t, history)
        loss = loss_fn(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader)}")

torch.save(model.state_dict(), "diffusion_handover_model.pth")
print("Model Saved!")