"""
timegrad_model.py
-----------------
TimeGrad: Autoregressive Diffusion Model for RSRP Forecasting.

Architecture: identical residual denoiser as RSRPDiffusion.

Key difference from DDPM
─────────────────────────
  DDPM context   : GRU(history)[-1]               one shared vector for all pred steps
  TimeGrad ctx_i : GRU(history || noisy[0..i-1])[seq_len-1+i]  per-step, grows autoregressively

Training  : forward(x_future_noisy, t, x_history)  ← same signature as RSRPDiffusion
Inference : denoise_with_state() called step-by-step from ddim_sampler.timegrad_sample()
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Identical to RSRPDiffusion.ResidualBlock."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.layer(x)


class TimeGradModel(nn.Module):
    """
    TimeGrad for RSRP time-series prediction.

    Training (teacher-forcing, efficient)
    ──────────────────────────────────────
    context_seq = [history (seq_len) | noisy_future[0..pred_len-2]]
    h_all = GRU(context_seq)                       # one GRU pass
    ctx_i = h_all[:, seq_len - 1 + i, :]          # per-step hidden state
    loss  = MSE(denoise(noisy_future[i], t, ctx_i), noise[i])

    All pred_len predictions happen in one forward pass — same training
    cost as DDPM.

    Inference (autoregressive, called from ddim_sampler.timegrad_sample)
    ──────────────────────────────────────────────────────────────────────
    h_state = encode(history)
    for i in range(pred_len):
        x[i]    = DDPM_reverse_loop(noise, h_state)   # uses denoise_with_state()
        h_state = GRU_update(x[i], h_state)            # context grows step by step
    """

    def __init__(self, input_size=1, hidden_dim=64, context_dim=32, num_layers=3):
        super().__init__()
        self.context_dim = context_dim

        # ── same components as RSRPDiffusion ──────────────────────────────────
        self.context_rnn  = nn.GRU(input_size, context_dim, batch_first=True)

        self.time_mlp     = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.input_proj   = nn.Linear(input_size, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        self.res_blocks   = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_layers)]
        )
        self.output_proj  = nn.Linear(hidden_dim, input_size)

    # ── Training forward (teacher-forcing) ───────────────────────────────────
    def forward(self, x_future_noisy, t, x_history):
        """
        Parameters
        ----------
        x_future_noisy : (B, pred_len, 1)   noisy target sequence
        t              : (B,)               diffusion timestep (same for all steps)
        x_history      : (B, seq_len, 1)    past RSRP context

        Returns
        -------
        predicted noise  (B, pred_len, 1)
        """
        B, pred_len, _ = x_future_noisy.shape

        # Build per-step context:
        # step 0  → context = GRU(history)[-1]
        # step i  → context = GRU(history + noisy_future[0..i-1])[-1]
        # Efficient: run GRU over [history | noisy_future[:-1]] in ONE pass,
        # then slice the last pred_len hidden states.
        shifted  = x_future_noisy[:, :-1, :]                          # (B, pred_len-1, 1)
        ctx_seq  = torch.cat([x_history, shifted], dim=1)             # (B, seq_len+pred_len-1, 1)
        h_all, _ = self.context_rnn(ctx_seq)                          # (B, seq_len+pred_len-1, ctx_dim)
        ctx      = h_all[:, -pred_len:, :]                            # (B, pred_len, ctx_dim)

        t_emb = self.time_mlp(t.float().view(-1, 1))                  # (B, hidden_dim)
        x     = self.input_proj(x_future_noisy)                       # (B, pred_len, hidden_dim)
        c     = self.context_proj(ctx)                                 # (B, pred_len, hidden_dim)

        h = x + t_emb.unsqueeze(1) + c
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)                                     # (B, pred_len, 1)

    # ── Inference helpers (used by ddim_sampler.timegrad_sample) ─────────────
    def denoise_with_state(self, x_noisy_step, t, h_state):
        """
        Denoise a single future step given the current GRU hidden state.
        Called repeatedly during autoregressive inference.

        Parameters
        ----------
        x_noisy_step : (B, 1, 1)            noisy scalar at this future step
        t            : (B,)                 diffusion timestep
        h_state      : (1, B, context_dim)  current GRU hidden state

        Returns
        -------
        predicted noise  (B, 1, 1)
        """
        context = h_state[-1]                                # (B, context_dim)
        t_emb   = self.time_mlp(t.float().view(-1, 1))      # (B, hidden_dim)
        x       = self.input_proj(x_noisy_step)             # (B, 1, hidden_dim)
        c       = self.context_proj(context)                 # (B, hidden_dim)
        h       = x + t_emb.unsqueeze(1) + c.unsqueeze(1)
        for block in self.res_blocks:
            h = block(h)
        return self.output_proj(h)                           # (B, 1, 1)
