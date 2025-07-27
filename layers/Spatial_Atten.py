import torch
import torch.nn as nn
import pickle

def load_adj_tensor(path: str) -> torch.BoolTensor:
    """
    从 pickle 文件加载稀疏邻接矩阵，转换为 PyTorch 布尔掩码张量
    """
    with open(path, 'rb') as f:
        A_sp = pickle.load(f)  # scipy.sparse.csr_matrix
    dense = A_sp.toarray().astype(bool)
    return torch.from_numpy(dense)

class SpatialAttention(nn.Module):
    """
    根据 Eq.(5)(6) 计算空间注意力矩阵 A ∈ R[n_s*n_s]:
      E = V · sigmoid((H' w1) W2 (w3 H')^T + B)
      A_i,j = softmax(E_i,j)  （仅在相邻路段对上计算）
    输入：
      H_prime: Tensor of shape [n_s, C, n_h] —— 段级时间特征 H'
    输出：
      A: Tensor of shape [n_s, n_s], 注意力权重矩阵
    """
    def __init__(self, n_s, n_h, C, adj_mask):
        super().__init__()
        # 参数矩阵
        self.w1 = nn.Parameter(torch.randn(C))             # R^C
        self.W2 = nn.Linear(n_h, C, bias=False)            # R^{n_h × C}
        self.w3 = nn.Parameter(torch.randn(n_h))           # R^{n_h}
        self.V  = nn.Parameter(torch.randn(n_s, n_s))      # R^{n_s × n_s}
        self.B  = nn.Parameter(torch.zeros(n_s, n_s))      # 偏置
        self.register_buffer('adj_mask', load_adj_tensor(adj_mask))  # 邻接矩阵掩码，形状 [n_s, n_s]，0/1 矩阵

    def forward(self, H_prime):
        # H_prime: [n_s, C, n_h]
        #把 [n_s, C, n_h] -> [n_s, n_h, C] [26659, 30, 16]
        H = H_prime.permute(0, 2, 1)

        # H′ w1 -> [n_s, n_h]，再乘 W2 -> [n_s, C]
        # w3 H′ -> [n_s, n_h]
        X1 = self.W2((H * self.w1).sum(dim=2))        # [n_s, C]
        X2 = (H * self.w3.view(1, -1, 1)).sum(dim=1)  # [n_s, C]

        # 先做外积，然后加偏置并过 σ
        E = self.V * torch.sigmoid(X1 @ X2.transpose(0, 1) + self.B)  # [n_s, n_s]

        # 对每个 i 只在其邻居 j∈N(i) 上做 softmax
        E_exp = torch.exp(E) * self.adj_mask  # adj_mask∈{0,1}遮掉非邻居
        A = E_exp / (E_exp.sum(dim=1, keepdim=True) + 1e-6)
        return A  # [n_s, n_s]