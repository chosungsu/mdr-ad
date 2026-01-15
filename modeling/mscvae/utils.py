import torch

def attribute_matrix(x):
    """
    x: (batch, seq, dim)
    return: (batch, seq, dim, dim) attribute matrices
    """
    batch, seq, dim = x.shape
    M = []
    for t in range(seq):
        xt = x[:, t]  # (batch, dim)
        # outer product for pairwise correlation
        mat = xt.unsqueeze(2) * xt.unsqueeze(1)
        M.append(mat)
    M = torch.stack(M, dim=1)  # (batch, seq, dim, dim)
    return M
