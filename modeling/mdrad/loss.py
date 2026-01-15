from __future__ import annotations

import torch
import torch.nn.functional as F


def mdrad_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    m_avg: torch.Tensor,
    m_hat: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
    *,
    alpha_corr: float = 1.0,
    beta_disc: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    - raw reconstruction (TCAD view)
    - correlation reconstruction (MSCVAE view, 평균 attribute matrix)
    - discrepancy penalty (GLAD/TCAD 스타일: latent 불일치)
    """
    recon_x = F.mse_loss(x_hat, x)
    recon_m = F.mse_loss(m_hat, m_avg)

    disc12 = F.mse_loss(z1, z2)
    disc13 = F.mse_loss(z1, z3)
    disc23 = F.mse_loss(z2, z3)
    disc = (disc12 + disc13 + disc23) / 3.0

    total = recon_x + alpha_corr * recon_m + beta_disc * disc
    return total, recon_x, recon_m, disc


def mdrad_anomaly_score(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    m_avg: torch.Tensor,
    m_hat: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
    z3: torch.Tensor,
    *,
    alpha_corr: float = 1.0,
    beta_disc: float = 0.1,
) -> torch.Tensor:
    """
    sample-wise anomaly score: (batch,)
    """
    # per-sample raw recon error
    recon_x = torch.mean((x_hat - x) ** 2, dim=(1, 2))
    recon_m = torch.mean((m_hat - m_avg) ** 2, dim=(1, 2))
    disc = (
        torch.mean((z1 - z2) ** 2, dim=1)
        + torch.mean((z1 - z3) ** 2, dim=1)
        + torch.mean((z2 - z3) ** 2, dim=1)
    ) / 3.0

    return recon_x + alpha_corr * recon_m + beta_disc * disc

