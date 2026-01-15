import torch
import torch.nn.functional as F
from utils import attribute_matrix


def mscvae_loss(x, recon_temporal, mus, logvars, kl_weight=0.1):
    """
    MSCVAE loss function
    
    Args:
        x: 원본 입력 (batch, seq, dim)
        recon_temporal: 재구성된 attribute matrix (batch, seq, 1, dim, dim)
        mus: 각 시점의 mu (list of (batch, latent_dim))
        logvars: 각 시점의 logvar (list of (batch, latent_dim))
        kl_weight: KL divergence 가중치
    
    Returns:
        total_loss: 전체 loss
    """
    # 원본 attribute matrix 계산
    M = attribute_matrix(x)  # (batch, seq, dim, dim)
    M = M.unsqueeze(2)  # (batch, seq, 1, dim, dim)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_temporal, M)
    
    # KL divergence loss (모든 시점의 평균)
    kl_loss = 0.0
    for mu, logvar in zip(mus, logvars):
        kl_loss += -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_loss = kl_loss / len(mus)
    
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
