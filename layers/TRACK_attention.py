import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_add, scatter_max

# ================= StaticSparseGAT（从头实现） =================
class StaticSparseGAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.d_model
        self.nh = args.n_heads
        assert d % self.nh == 0, "d_model must be divisible by n_heads"
        self.hd = d // self.nh
        self.d = d

        # W1, W2 project node features to d (same dim as hidden)
        self.W1 = nn.Linear(d, d, bias=False)
        self.W2 = nn.Linear(d, d, bias=False)
        # W3 projects scalar P_edge to d-dim vector before W4
        self.W3 = nn.Linear(1, d, bias=False)
        # W4 maps d -> nh (per-head logit)
        self.W4 = nn.Linear(d, self.nh, bias=False)

        # value projection and output
        self.Wv = nn.Linear(d, d, bias=False)  # project src to value-space
        self.Wout = nn.Linear(d, d)
        self.res = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(args.dropout if hasattr(args, 'dropout') else 0.1)

    def forward(self, H, edge_index, P_edge=None, deter_edge=None):
        """
        H: N x d
        edge_index: 2 x E (src,dst)
        P_edge: E or None  (log-prob p_{i,j,t})
        deter_edge: E or None (extra scalar to add to logits)
        returns: N x d
        """
        device = H.device
        src = edge_index[0]
        dst = edge_index[1]
        N = H.size(0)
        E = src.size(0)

        # node projections
        H_W1 = self.W1(H)  # N x d
        H_W2 = self.W2(H)  # N x d
        V = self.Wv(H)     # N x d (values)

        # gather per-edge
        h_i = H_W1[dst]  # E x d  (note: dst is target node i)
        h_j = H_W2[src]  # E x d  (source node j)
        # base sum: E x d
        e_base = h_i + h_j

        # add P_edge contribution if present
        if P_edge is not None:
            # project scalar to d-dim and add
            p_vec = self.W3(P_edge.view(-1, 1).to(device))  # E x d
            e_base = e_base + p_vec

        # compute logits per head: E x nh
        logits = self.W4(e_base)  # E x nh

        # add deterrence if present (broadcast to heads)
        if deter_edge is not None:
            logits = logits + deter_edge.view(-1, 1).to(device)

        # LeakyReLU (per paper)
        logits = self.leaky(logits)  # E x nh

        # segmented softmax grouped by dst, per head
        logits_t = logits.permute(1, 0)  # nh x E
        alphas_list = []
        for h in range(self.nh):
            sh = logits_t[h]  # E
            max_per_dst = scatter_max(sh, dst, dim=0)[0]  # N
            max_e = max_per_dst[dst]  # E
            exp_e = (sh - max_e).exp()
            sum_per_dst = scatter_add(exp_e, dst, dim=0)  # N
            alpha_e = exp_e / (sum_per_dst[dst] + 1e-12)
            alphas_list.append(alpha_e)
        alpha = torch.stack(alphas_list, dim=1)  # E x nh
        # dropout on attention
        alpha = self.dropout(alpha)  # E x nh

        # compute value vectors per edge: gather V[src] and split into heads
        v_e = V[src].view(E, self.nh, self.hd)  # E x nh x hd

        # weighted sum
        messages = alpha.unsqueeze(-1) * v_e  # E x nh x hd

        # aggregate per dst
        msgs_flat = messages.view(E, self.nh * self.hd)  # E x d
        if self.use_scatter:
            agg_flat = scatter_add(msgs_flat, dst, dim=0, dim_size=N)  # N x d
        else:
            agg_flat = torch.zeros((N, self.nh * self.hd), device=device)
            agg_flat = agg_flat.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.nh * self.hd), msgs_flat)
        out = agg_flat.view(N, self.nh * self.hd)  # N x d

        out = self.Wout(out)
        out = self.ln(out + self.res(H))
        return out

# ---------------- Sparse Paper-style Cross GAT ----------------
class CrossSparseGAT(nn.Module):
    """
    Cross-view paper-style sparse GAT (add-then-project form) — 对应论文公式 (11)/(12)。

    语义：
      - dst_feats: N_dst x d  (queries 所在视图的节点表示)
      - src_feats: N_src x d  (keys/values 所在视图的节点表示)
      - edge_index: 2 x E, (src_idx, dst_idx) 表示跨视图的边（消息从 src -> dst）
      - P_edge: optional E 标量（例如 p_{i,j,t}）会被映射为向量并加入到 score 的加法项
      - deter_edge: optional E 标量（几何衰减项），直接作为 logit 的偏置（广播到 heads）
    实现：
      e_{i,j} = (h_i W1 + h_j W2 + p_{ij} W3) W4^T
      alpha = softmax_dst( LeakyReLU(e_{i,j} + deter_{ij}) )
      out_dst = LayerNorm( Wout( Σ_j alpha_{ij} * V(h_j) ) + Res(h_i) )
    对应论文：公式 (11)/(12)（Cross co-attention）。
    """
    def __init__(self, args):
        super().__init__()
        d = args.d_model
        self.nh = args.n_heads
        assert d % self.nh == 0, "d_model must be divisible by n_heads"
        self.hd = d // self.nh
        self.d = d

        # 按论文符号分解：W1（dst 投影）、W2（src 投影）、W3（边标量投影）、W4（投到 nh logits）
        self.W1 = nn.Linear(d, d, bias=False)   # dst node linear proj (h_i W1)
        self.W2 = nn.Linear(d, d, bias=False)   # src node linear proj (h_j W2)
        self.W3 = nn.Linear(1, d, bias=False)   # project scalar P_edge -> d-vector
        self.W4 = nn.Linear(d, self.nh, bias=False)  # map d -> nh (per-head logits)

        # value projection & output/residual/norm
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wout = nn.Linear(d, d)
        self.res = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(args.dropout if hasattr(args, 'dropout') else 0.1)


    def forward(self, dst_feats, src_feats, edge_index, P_edge=None, deter_edge=None):
        """
        dst_feats: N_dst x d
        src_feats: N_src x d
        edge_index: 2 x E (src_idx, dst_idx), src_idx in [0..N_src-1], dst_idx in [0..N_dst-1]
        P_edge: optional E tensor (标量，如 p_{i,j,t} 或 logp)
        deter_edge: optional E tensor (几何衰减项)
        returns: N_dst x d  (更新后的 dst 视图节点表示)
        """
        device = dst_feats.device
        src_idx = edge_index[0]   # E
        dst_idx = edge_index[1]   # E
        N_dst = dst_feats.size(0)
        E = src_idx.size(0)

        # 1) 节点线性投影（到 d 维，再由 W4 投到每个 head 的 logit）
        dst_proj = self.W1(dst_feats)   # N_dst x d  (h_i W1)
        src_proj = self.W2(src_feats)   # N_src x d  (h_j W2)
        V = self.Wv(src_feats)          # N_src x d  (value projection)

        # 2) gather per-edge 的 dst/src 投影并相加
        h_i = dst_proj[dst_idx]   # E x d  (dst for each edge)
        h_j = src_proj[src_idx]   # E x d  (src for each edge)
        e_base = h_i + h_j        # E x d

        # 3) 如果有 P_edge（例如 traj 的 p_{i,j,t}）则把它投影为 d 维并加到基向量上
        if P_edge is not None:
            p_vec = self.W3(P_edge.view(-1, 1).to(device))  # E x d
            e_base = e_base + p_vec

        # 4) 把 e_base 投影为每 head 的 logits（E x nh）
        logits = self.W4(e_base)  # E x nh

        # 5) 如果有 deter_edge（几何衰减项），在 logits 上直接加偏置（广播到 head 维度）
        if deter_edge is not None:
            logits = logits + deter_edge.view(-1, 1).to(device)

        # 6) LeakyReLU（如论文）
        logits = self.leaky(logits)  # E x nh

        # 7) segmented softmax：按每个 dst 聚合（对每个 head 单独做）
        logits_t = logits.permute(1, 0)  # nh x E
        alphas_list = []
        for h in range(self.nh):
            sh = logits_t[h]  # E
            max_per_dst = scatter_max(sh, dst_idx, dim=0)[0]  # N_dst
            max_e = max_per_dst[dst_idx]  # E
            exp_e = (sh - max_e).exp()
            sum_per_dst = scatter_add(exp_e, dst_idx, dim=0)  # N_dst
            alpha_e = exp_e / (sum_per_dst[dst_idx] + 1e-12)
            alphas_list.append(alpha_e)
        alpha = torch.stack(alphas_list, dim=1)  # E x nh

        # 8) dropout
        alpha = self.dropout(alpha)  # E x nh

        # 9) gather value vectors并按 head 分割，计算消息
        v_e = V[src_idx].view(E, self.nh, self.hd)  # E x nh x hd
        messages = alpha.unsqueeze(-1) * v_e         # E x nh x hd

        # 10) aggregate messages 到 dst nodes
        msgs_flat = messages.view(E, self.nh * self.hd)  # E x d
        if self.use_scatter:
            agg_flat = scatter_add(msgs_flat, dst_idx, dim=0, dim_size=N_dst)  # N_dst x d
        else:
            agg_flat = torch.zeros((N_dst, self.nh * self.hd), device=device)
            agg_flat = agg_flat.scatter_add_(0, dst_idx.unsqueeze(-1).expand(-1, self.nh * self.hd), msgs_flat)
        out = agg_flat.view(N_dst, self.nh * self.hd)  # N_dst x d

        # 11) 输出投影 + 残差 + LayerNorm
        out = self.Wout(out)
        out = self.ln(out + self.res(dst_feats))
        return out
    
# ---------------- Co-Attentional Layer（严格实现） ----------------
class CoAttLayer(nn.Module):
    """
    严格的 co-attentional block：
      - traj_from_traf: traj <- traf (对应论文 (11) 中 traj query)
      - traf_from_traj: traf <- traj (对应论文 (12) 中 traf query)
      每个子层后接残差 + LayerNorm + FFN
    """
    def __init__(self, args):
        super().__init__()
        self.traj_from_traf = CrossSparseGAT(args)  # traj queries, traf keys/values
        self.traf_from_traj = CrossSparseGAT(args)  # traf queries, traj keys/values

        self.norm_traj1 = nn.LayerNorm(args.d_model)
        self.norm_traj2 = nn.LayerNorm(args.d_model)
        self.norm_traf1 = nn.LayerNorm(args.d_model)
        self.norm_traf2 = nn.LayerNorm(args.d_model)

        self.ffn_traj = nn.Sequential(nn.Linear(args.d_model, args.d_model*4),
                                      nn.ReLU(),
                                      nn.Linear(args.d_model*4, args.d_model),
                                      nn.Dropout(0.1))
        self.ffn_traf = nn.Sequential(nn.Linear(args.d_model, args.d_model*4),
                                      nn.ReLU(),
                                      nn.Linear(args.d_model*4, args.d_model),
                                      nn.Dropout(0.1))

    def forward(self, H_traj, H_traf, edge_index_traf2traj, edge_index_traj2traf,
                P_edge_traf2traj=None, P_edge_traj2traf=None, deter_traf2traj=None, deter_traj2traf=None):
        # traj <- traf (公式 11)
        cross_traj = self.traj_from_traf(H_traj, H_traf, edge_index_traj2traf, P_edge=P_edge_traj2traf, deter_edge=deter_traj2traf)
        H_traj = self.norm_traj1(H_traj + cross_traj)
        H_traj = self.norm_traj2(H_traj + self.ffn_traj(H_traj))

        # traf <- traj (公式 12)
        cross_traf = self.traf_from_traj(H_traf, H_traj, edge_index_traf2traj, P_edge=P_edge_traf2traj, deter_edge=deter_traf2traj)
        H_traf = self.norm_traf1(H_traf + cross_traf)
        H_traf = self.norm_traf2(H_traf + self.ffn_traf(H_traf))

        return H_traj, H_traf
