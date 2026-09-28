import torch
import torch.nn as nn
import math


# --------------------------------------------------
# Sinusoidal Time Embedding (IMPORTANT UPGRADE)
# --------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )

        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()),
            dim=-1
        )

        return embeddings


# --------------------------------------------------
# Residual Block
# --------------------------------------------------
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


# --------------------------------------------------
# Main Diffusion Model
# --------------------------------------------------
class RSRPDiffusion(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_dim=128,
        context_dim=64,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        # ------------------------------------------
        # 1. Context Encoder (History Encoder)
        # ------------------------------------------
        self.context_rnn = nn.GRU(
            input_size,
            context_dim,
            batch_first=True
        )

        # ------------------------------------------
        # 2. Better Time Embedding
        # ------------------------------------------
        self.time_embedding = SinusoidalPositionEmbeddings(hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ------------------------------------------
        # 3. Input Projection
        # ------------------------------------------
        self.input_proj = nn.Linear(
            input_size,
            hidden_dim
        )

        self.context_proj = nn.Linear(
            context_dim,
            hidden_dim
        )

        # ------------------------------------------
        # 4. Residual Denoising Blocks
        # ------------------------------------------
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                hidden_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # ------------------------------------------
        # 5. Final Output Layer
        # ------------------------------------------
        self.output_proj = nn.Linear(
            hidden_dim,
            input_size
        )

    def forward(
        self,
        x_future_noisy,
        t,
        x_history
    ):
        """
        x_future_noisy:
            [B, pred_len, 1]

        t:
            [B]

        x_history:
            [B, seq_len, 1]
        """

        # ------------------------------------------
        # Encode history
        # ------------------------------------------
        _, h_n = self.context_rnn(x_history)

        context = h_n[-1]  # [B, context_dim]

        # ------------------------------------------
        # Better timestep embedding
        # ------------------------------------------
        t_emb = self.time_embedding(
            t.float()
        )

        t_emb = self.time_mlp(
            t_emb
        )  # [B, hidden_dim]

        # ------------------------------------------
        # Project inputs
        # ------------------------------------------
        x = self.input_proj(
            x_future_noisy
        )

        c = self.context_proj(
            context
        )

        # ------------------------------------------
        # Combine
        # ------------------------------------------
        h = (
            x
            + t_emb.unsqueeze(1)
            + c.unsqueeze(1)
        )

        # ------------------------------------------
        # Residual denoising
        # ------------------------------------------
        for block in self.res_blocks:
            h = block(h)

        # ------------------------------------------
        # Predict noise
        # ------------------------------------------
        out = self.output_proj(h)

        return out