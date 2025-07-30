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

def graph_propagation_sparse_batch(x, A, hop=10, dual=False):
    """
    x:   Tensor of shape (B, N)
    A:   either
           - sparse Tensor of shape (N, N), shared for all B, or
           - dense Tensor of shape (B, N, N)
    hop: number of propagation steps
    dual: whether 做双向 random‑walk

    returns: Tensor of shape
           - non‑dual: (B, N, hop+1)
           - dual:     (B, N, 1 + 2*hop)
    """
    B, N = x.shape

    # 如果 A 是稀疏且只有两维，就视为共享邻接，先扩成 dense batch 形式
    if A.is_sparse and A.dim() == 2:
        # 稀疏 -> dense
        A_dense = A.to_dense().unsqueeze(0).expand(B, N, N).contiguous()
    elif not A.is_sparse and A.dim() == 3 and A.shape[0] == B:
        A_dense = A
    else:
        raise ValueError("A must be either sparse (N,N) or dense (B,N,N)")

    # 初始 y: 从 (B,N) 变成 (B,N,1)
    y = x.unsqueeze(2)
    X = y

    if dual:
        # 双向传播：每步拼 downstream+upstream
        A_up = A_dense.transpose(1, 2)
        for _ in range(hop):
            y_down = torch.bmm(A_dense, y)  # (B,N,1)
            y_up   = torch.bmm(A_up,    y)  # (B,N,1)
            # concat [原始, 下游, 上游]
            X = torch.cat([X, y_down, y_up], dim=2)
        # 结果形状 (B, N, 1 + 2*hop)
    else:
        # 单向传播：每步拼新的 y
        for _ in range(hop):
            y = torch.bmm(A_dense, y)      # (B,N,1)
            X = torch.cat([X, y], dim=2)
        # 结果形状 (B, N, hop+1)

    return X