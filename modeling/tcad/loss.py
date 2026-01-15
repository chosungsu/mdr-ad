import torch.nn.functional as F

def tcad_loss(x, x_hat, z1, z2):
    return F.mse_loss(x_hat, x) + F.mse_loss(z1, z2)
