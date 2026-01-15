import torch


def attribute_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    x: (batch, seq, dim)
    return: (batch, seq, dim, dim) attribute matrices (outer products)
    """
    batch, seq, dim = x.shape
    mats = []
    for t in range(seq):
        xt = x[:, t]  # (batch, dim)
        mats.append(xt.unsqueeze(2) * xt.unsqueeze(1))
    return torch.stack(mats, dim=1)

