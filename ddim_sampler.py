"""
ddim_sampler.py
---------------
Shared sampling functions imported by every inference script.
Eliminates the repeated inline DDPM loops scattered across the codebase.

Exports
-------
build_noise_schedule(device)          -> betas, alphas, alphas_cumprod
ddpm_sample(...)                      -> x, elapsed_s
ddim_sample(...)                      -> x, elapsed_s   (50 steps default)
timegrad_sample(...)                  -> x, elapsed_s   (autoregressive)
denorm(x_norm, rsrp_min, rsrp_max)   -> dBm values
"""

import torch
import time

# Must match train_diffusion.py / train.py exactly
TIMESTEPS  = 100
BETA_START = 1e-4
BETA_END   = 0.02


# ── Noise schedule ────────────────────────────────────────────────────────────
def build_noise_schedule(device):
    """Returns (betas, alphas, alphas_cumprod) on the given device."""
    betas          = torch.linspace(BETA_START, BETA_END, TIMESTEPS, device=device)
    alphas         = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


# ── DDPM sampler (original, 100 steps) ───────────────────────────────────────
@torch.no_grad()
def ddpm_sample(model, history, n_samples, pred_len,
                betas, alphas, alphas_cumprod, device):
    """
    Standard DDPM: all pred_len steps denoised in parallel over 100 steps.

    history   : (1, seq_len, 1) – single context window
    Returns   : (n_samples, pred_len, 1) in [-1,1], elapsed seconds
    """
    model.eval()
    h = history.expand(n_samples, -1, -1).contiguous()

    x  = torch.randn(n_samples, pred_len, 1, device=device)
    t0 = time.perf_counter()

    for t in reversed(range(TIMESTEPS)):
        t_t   = torch.full((n_samples,), t, device=device, dtype=torch.long)
        eps   = model(x, t_t, h)
        alpha = alphas[t]
        abar  = alphas_cumprod[t]
        beta  = betas[t]
        z     = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x     = (1.0 / alpha.sqrt()) * (
                    x - (1.0 - alpha) / (1.0 - abar).sqrt() * eps
                ) + beta.sqrt() * z

    return x, time.perf_counter() - t0


# ── DDIM sampler (50 steps, no retraining needed) ────────────────────────────
@torch.no_grad()
def ddim_sample(model, history, n_samples, pred_len,
                alphas_cumprod, ddim_steps=50, eta=0.0, device='cpu'):
    """
    DDIM: all pred_len steps denoised in parallel over `ddim_steps` steps.
    Uses the same trained DDPM weights — no retraining required.

    eta = 0.0  → fully deterministic (default, recommended)
    eta = 1.0  → recovers DDPM-like stochasticity

    Returns : (n_samples, pred_len, 1) in [-1,1], elapsed seconds
    """
    model.eval()
    h = history.expand(n_samples, -1, -1).contiguous()

    # Evenly spaced subset of timesteps: [99, 97, 95, ..., 1] for 50 steps
    step_ratio = TIMESTEPS // ddim_steps
    timesteps  = list(range(0, TIMESTEPS, step_ratio))[::-1]

    x  = torch.randn(n_samples, pred_len, 1, device=device)
    t0 = time.perf_counter()

    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i]
        t_prev = timesteps[i + 1]

        t_t        = torch.full((n_samples,), t_curr, device=device, dtype=torch.long)
        alpha_t    = alphas_cumprod[t_curr]
        alpha_prev = alphas_cumprod[t_prev]

        eps  = model(x, t_t, h)

        # Predict x_0, clamp for numerical stability
        x0   = ((x - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt()).clamp(-1.0, 1.0)

        if eta > 0.0:
            sigma  = (eta
                      * ((1.0 - alpha_prev) / (1.0 - alpha_t)).sqrt()
                      * (1.0 - alpha_t / alpha_prev).sqrt())
            dir_xt = (1.0 - alpha_prev - sigma ** 2).sqrt() * eps
            x      = alpha_prev.sqrt() * x0 + dir_xt + sigma * torch.randn_like(x)
        else:
            dir_xt = (1.0 - alpha_prev).sqrt() * eps
            x      = alpha_prev.sqrt() * x0 + dir_xt

    return x, time.perf_counter() - t0


# ── TimeGrad sampler (autoregressive, 100 steps per future step) ─────────────
@torch.no_grad()
def timegrad_sample(model, history, n_samples, pred_len,
                    betas, alphas, alphas_cumprod, device):
    """
    TimeGrad: generates pred_len future steps ONE-BY-ONE.
    Each step runs a full DDPM reverse loop (100 steps), conditioned on
    the current GRU hidden state which grows with every generated step.

    Key property: step i is conditioned on generated steps 0..i-1
                  → temporally coherent trajectories.

    Inference cost: pred_len × DDPM cost  (~10× slower than DDPM for pred_len=10).
    This cost difference is a primary result in the comparison.

    history   : (1, seq_len, 1)
    Returns   : (n_samples, pred_len, 1) in [-1,1], elapsed seconds
    """
    model.eval()
    h = history.expand(n_samples, -1, -1).contiguous()

    # Encode history into initial GRU hidden state
    _, h_state = model.context_rnn(h)           # (1, n_samples, context_dim)

    generated = []
    t0        = time.perf_counter()

    for _ in range(pred_len):
        # Fresh noise for this single future step
        x_step = torch.randn(n_samples, 1, 1, device=device)

        # Full DDPM reverse loop for ONE scalar step
        for t in reversed(range(TIMESTEPS)):
            t_t   = torch.full((n_samples,), t, device=device, dtype=torch.long)
            eps   = model.denoise_with_state(x_step, t_t, h_state)
            alpha = alphas[t]
            abar  = alphas_cumprod[t]
            beta  = betas[t]
            z     = torch.randn_like(x_step) if t > 0 else torch.zeros_like(x_step)
            x_step = (1.0 / alpha.sqrt()) * (
                         x_step - (1.0 - alpha) / (1.0 - abar).sqrt() * eps
                     ) + beta.sqrt() * z

        generated.append(x_step)

        # Update GRU with the generated step before producing the next one
        _, h_state = model.context_rnn(x_step, h_state)

    return torch.cat(generated, dim=1), time.perf_counter() - t0


# ── Denormalisation ───────────────────────────────────────────────────────────
def denorm(x_norm, rsrp_min, rsrp_max):
    """Convert model output in [-1, 1] back to dBm."""
    return ((x_norm + 1.0) / 2.0) * (rsrp_max - rsrp_min) + rsrp_min
