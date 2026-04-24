import torch
import torch.nn as nn
import math

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.layer(x)

class RSRPDiffusion(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64, context_dim=32, num_layers=3):
        super().__init__()
        
        # 1. Context Encoder (Reads past history)
        self.context_rnn = nn.GRU(input_size, context_dim, batch_first=True)
        
        # 2. Time Embedding (Tells model which diffusion step we are at)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 3. Denoising Network (Predicts noise)
        self.input_proj = nn.Linear(input_size, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, input_size)

    def forward(self, x_future_noisy, t, x_history):
        # Encode Past History
        _, h_n = self.context_rnn(x_history)
        context = h_n[-1] # Last hidden state [Batch, Context_Dim]
        
        # Embed Time
        t_emb = self.time_mlp(t.float().view(-1, 1))
        
        # Project Inputs
        x = self.input_proj(x_future_noisy)
        c = self.context_proj(context)
        
        # Combine Signal + Time + Context
        # Note: We broadcast context and time across the sequence if needed
        # For simplicity, we assume x_future is [Batch, Seq, Feat] and we process per-step or flatten.
        # Here we treat the sequence dimension as features for the linear layer or loop.
        # Simplified approach: Process the whole sequence window at once.
        
        h = x + t_emb.unsqueeze(1) + c.unsqueeze(1)
        
        for block in self.res_blocks:
            h = block(h)
            
        return self.output_proj(h)
