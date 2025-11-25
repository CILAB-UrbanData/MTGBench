# Spatial_Atten_batched.py

import torch
import torch.nn as nn
import pickle
import numpy as np


def load_adj_tensor(path_or_tensor):
    # 如果是 tensor 直接返回（转换成 bool）
    if isinstance(path_or_tensor, torch.Tensor):
        return path_or_tensor.bool()

    # 如果是 numpy 直接返回
    if isinstance(path_or_tensor, np.ndarray):
        return torch.from_numpy(path_or_tensor.astype(bool))

    # 如果是 str → 从 pkl 读
    if isinstance(path_or_tensor, str):
        with open(path_or_tensor, 'rb') as f:
            A = pickle.load(f)
        # 自动判断类型
        if isinstance(A, torch.Tensor):
            return A.bool()
        elif isinstance(A, np.ndarray):
            return torch.from_numpy(A.astype(bool))
        else:
            # 假设是 scipy.sparse
            return torch.from_numpy(A.toarray().astype(bool))

    raise TypeError



class SpatialAttention(nn.Module):
    """
    Batched 版本的空间注意力：

    输入:
      H_prime:
        - [N, C, n_h]  单样本
        - [B, N, C, n_h]  多样本

    输出:
      A:
        - [N, N]       （单样本）
        - [B, N, N]    （多样本）

    公式仍是你原来的：
      E = V ⊙ sigmoid( (H'w1)W2 (w3H')^T + B )
      然后对邻居做 softmax（用 adj_mask 屏蔽非邻居）
    """

    def __init__(self, NumofRoads, n_h, C, adj_mask):
        super().__init__()
        self.NumofRoads = NumofRoads
        self.n_h = n_h
        self.C   = C

        # 参数矩阵
        self.w1 = nn.Parameter(torch.randn(C))           # R^C
        self.W2 = nn.Linear(n_h, C, bias=False)          # R^{n_h -> C}
        self.w3 = nn.Parameter(torch.randn(n_h))         # R^{n_h}
        self.V  = nn.Parameter(torch.randn(NumofRoads, NumofRoads))    # R^{NumofRoads x NumofRoads}
        self.B  = nn.Parameter(torch.zeros(NumofRoads, NumofRoads))    # 偏置

        # 邻接掩码 [NumofRoads, NumofRoads]
        adj_bool = load_adj_tensor(adj_mask)
        assert adj_bool.shape == (NumofRoads, NumofRoads), \
            f"adj_mask shape {adj_bool.shape} != (NumofRoads={NumofRoads}, NumofRoads={NumofRoads})"
        self.register_buffer('adj_mask', adj_bool)

    def forward(self, H_prime: torch.Tensor) -> torch.Tensor:
        """
        H_prime:
          - [N, C, n_h]
          - [B, N, C, n_h]
        """
        single = False
        if H_prime.dim() == 3:
            # [N,C,n_h] -> [1,N,C,n_h]
            H_prime = H_prime.unsqueeze(0)
            single = True

        B, N, C, n_h = H_prime.shape
        assert N == self.NumofRoads and C == self.C and n_h == self.n_h, \
            f"H_prime shape {(B,N,C,n_h)} not match (NumofRoads={self.NumofRoads}, C={self.C}, n_h={self.n_h})"

        # H: [B,N,n_h,C]
        H = H_prime.permute(0, 1, 3, 2)

        # (H * w1) sum over C -> [B,N,n_h]
        # w1: [C]
        X1_tmp = (H * self.w1.view(1, 1, 1, C)).sum(dim=3)    # [B,N,n_h]

        # W2((H'w1)(...))：先展平 B*N，再线性映射到 C
        X1_flat = X1_tmp.reshape(B * N, n_h)                  # [B*N,n_h]
        X1_flat = self.W2(X1_flat)                            # [B*N,C]
        X1 = X1_flat.view(B, N, C)                            # [B,N,C]

        # (H * w3) sum over n_h -> [B,N,C]
        X2_tmp = (H * self.w3.view(1, 1, n_h, 1)).sum(dim=2)  # [B,N,C]
        X2 = X2_tmp                                           # [B,N,C]

        # pairwise 相似度: [B,N,C] @ [B,C,N] -> [B,N,N]
        S = torch.matmul(X1, X2.transpose(-1, -2))            # [B,N,N]

        # E = V ⊙ sigmoid(S + B)
        V = self.V.unsqueeze(0)       # [1,N,N]
        B_bias = self.B.unsqueeze(0)  # [1,N,N]
        E = V * torch.sigmoid(S + B_bias)  # [B,N,N]

        # 对每个 i 只在邻居上 softmax
        mask = self.adj_mask.unsqueeze(0)  # [1,N,N]
        E_exp = torch.exp(E) * mask        # 非邻居位置=0
        denom = E_exp.sum(dim=-1, keepdim=True) + 1e-6
        A = E_exp / denom                  # [B,N,N]

        if single:
            return A[0]                    # [N,N]
        return A                           # [B,N,N]
