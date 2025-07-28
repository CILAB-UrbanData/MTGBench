import torch

def gcn_norm(A):
    # A: [batch, N, N]
    I = torch.eye(A.size(-1), device=A.device)
    A_hat = A + I
    D = torch.sum(A_hat, dim=-1)
    D_inv_sqrt = torch.diag_embed(D.pow(-0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

def graph_propagation_sparse(x, A, hop=10, dual=False):
    # sparse version
    # x: graph signal vector. tensor. (n_road)
    # A: adjacency matrix. tranposed. sparse_tensor. (n_road, n_road)
    # hop: # propagation steps
    # output: propagation result. tensor. (n_road, hop+1)
    
    y = x.unsqueeze(1)
    X = y
    if dual: # dual random walk
        for i in range(hop):
            y_down = A.mm(X) # downstream
            y_up = A.transpose(0, 1).mm(X) # upstream
            X = torch.cat([y, y_down, y_up], dim=1)
    else: # downstream random walk only
        for i in range(hop):
            y = A.mm(y)
            X = torch.cat([X, y], dim=1)
    return X