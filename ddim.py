import numpy as np
import torch
from typing import Dict, Optional


def build_schedule(
    timesteps: int = 100,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Create the beta/alpha schedule used for both DDPM and DDIM."""
    device = device or torch.device("cpu")
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {"betas": betas, "alphas": alphas, "alphas_cumprod": alphas_cumprod}


def denormalize_rsrp(x_norm: np.ndarray, rsrp_min: float, rsrp_max: float) -> np.ndarray:
    return ((x_norm + 1.0) / 2.0) * (rsrp_max - rsrp_min) + rsrp_min


def normalize_rsrp(x_dbm: np.ndarray, rsrp_min: float, rsrp_max: float) -> np.ndarray:
    return (x_dbm - rsrp_min) / (rsrp_max - rsrp_min) * 2.0 - 1.0


def get_ddim_timesteps(num_training_steps: int, num_inference_steps: int, device: torch.device) -> torch.Tensor:
    """Return a descending list of timesteps for DDIM."""
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    num_inference_steps = min(num_inference_steps, num_training_steps)
    timesteps = np.linspace(0, num_training_steps - 1, num_inference_steps)
    timesteps = np.round(timesteps).astype(np.int64)
    timesteps = np.unique(timesteps)[::-1].copy()
    return torch.tensor(timesteps, dtype=torch.long, device=device)


@torch.no_grad()
def sample_ddpm(
    model,
    history: torch.Tensor,
    future_len: int,
    schedule: Dict[str, torch.Tensor],
    n_samples: int = 50,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Standard stochastic reverse diffusion (DDPM)."""
    device = device or history.device
    if history.shape[0] != 1:
        raise ValueError("sample_ddpm currently expects history batch size = 1")
    model.eval()

    betas = schedule["betas"]
    alphas = schedule["alphas"]
    alphas_cumprod = schedule["alphas_cumprod"]

    history_expanded = history.repeat(n_samples, 1, 1)
    x = torch.randn(n_samples, future_len, 1, device=device, generator=generator)

    for t in reversed(range(len(betas))):
        t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
        pred_noise = model(x, t_tensor, history_expanded)

        alpha_t = alphas[t]
        alpha_bar_t = alphas_cumprod[t]
        beta_t = betas[t]

        noise = torch.randn(x.shape, device=device, generator=generator) if t > 0 else torch.zeros_like(x)

        x = (1.0 / torch.sqrt(alpha_t)) * (
            x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        ) + torch.sqrt(beta_t) * noise

    return x


@torch.no_grad()
def sample_ddim(
    model,
    history: torch.Tensor,
    future_len: int,
    schedule: Dict[str, torch.Tensor],
    n_samples: int = 50,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """DDIM sampler for time-series tensors [B, future_len, 1]."""
    device = device or history.device
    if history.shape[0] != 1:
        raise ValueError("sample_ddim currently expects history batch size = 1")
    model.eval()

    betas = schedule["betas"]
    alphas_cumprod = schedule["alphas_cumprod"]
    timesteps = get_ddim_timesteps(len(betas), num_inference_steps, device)

    history_expanded = history.repeat(n_samples, 1, 1)
    x = torch.randn(n_samples, future_len, 1, device=device, generator=generator)

    for i, t in enumerate(timesteps):
        t_int = int(t.item())
        t_prev = int(timesteps[i + 1].item()) if i < len(timesteps) - 1 else -1

        t_batch = torch.full((n_samples,), t_int, device=device, dtype=torch.long)
        pred_noise = model(x, t_batch, history_expanded)

        alpha_bar_t = alphas_cumprod[t_int].clamp(min=1e-8, max=1.0).view(1, 1, 1)
        alpha_bar_prev = (
            alphas_cumprod[t_prev].clamp(min=1e-8, max=1.0).view(1, 1, 1)
            if t_prev >= 0
            else torch.ones(1, 1, 1, device=device)
        )

        x0_pred = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        if eta > 0 and t_prev >= 0:
            sigma_t = eta * torch.sqrt(
                torch.clamp(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_prev),
                    min=0.0,
                )
            )
            noise = torch.randn(x.shape, device=device, generator=generator)
        else:
            sigma_t = torch.zeros(1, 1, 1, device=device)
            noise = torch.zeros_like(x)

        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * pred_noise
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma_t * noise

    return x


def sample_futures(
    model,
    history: torch.Tensor,
    future_len: int,
    sampler: str,
    schedule: Dict[str, torch.Tensor],
    n_samples: int = 50,
    num_inference_steps: int = 50,
    eta: float = 0.0,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    sampler = sampler.lower().strip()
    if sampler == "ddpm":
        return sample_ddpm(model, history, future_len, schedule, n_samples, device, generator)
    if sampler == "ddim":
        return sample_ddim(
            model,
            history,
            future_len,
            schedule,
            n_samples=n_samples,
            num_inference_steps=num_inference_steps,
            eta=eta,
            device=device,
            generator=generator,
        )
    raise ValueError("sampler must be one of: 'ddpm', 'ddim'")


def calculate_crps_ensemble(forecasts: np.ndarray, observation: np.ndarray) -> float:
    forecasts = np.asarray(forecasts, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    term1 = np.mean(np.abs(forecasts - observation))
    pairwise = np.abs(forecasts[:, None, :] - forecasts[None, :, :]).mean()
    return float(term1 - 0.5 * pairwise)


def evaluate_failure_probability(forecasts_dbm: np.ndarray, threshold_dbm: float) -> float:
    return float(np.mean(np.min(forecasts_dbm, axis=1) < threshold_dbm))
