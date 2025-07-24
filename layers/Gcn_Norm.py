import torch

def gcn_norm(A):
    # A: [batch, N, N]
    I = torch.eye(A.size(-1), device=A.device)
    A_hat = A + I
    D = torch.sum(A_hat, dim=-1)
    D_inv_sqrt = torch.diag_embed(D.pow(-0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt