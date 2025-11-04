import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_max


# ============== 工具 ==============

def _is_batched(x):
    return x.dim() == 3  # (B,N,d)

def _has_batched_edges(edge_index):
    return edge_index.dim() == 3  # (B,2,E)

def _maybe_flatten_edge_attr(attr, B, E, device):
    if attr is None:
        return None
    if attr.dim() == 1:           # (E,)
        return attr.to(device).reshape(-1)           # (E,)
    elif attr.dim() == 2:         # (B,E)
        return attr.to(device).reshape(B * E)        # (B*E,)
    else:
        raise ValueError("P_edge/deter_edge 形状需为 (E,) 或 (B,E)")


# ============== StaticSparseGAT（支持单图/Batch） ==============
class StaticSparseGAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.d_model
        self.nh = args.n_heads
        assert d % self.nh == 0, "d_model must be divisible by n_heads"
        self.hd = d // self.nh
        self.d = d

        self.W1 = nn.Linear(d, d, bias=False)
        self.W2 = nn.Linear(d, d, bias=False)
        self.W3 = nn.Linear(1, d, bias=False)
        self.W4 = nn.Linear(d, self.nh, bias=False)

        self.Wv = nn.Linear(d, d, bias=False)
        self.Wout = nn.Linear(d, d)
        self.res = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.1))
        self.use_scatter = True  # 默认使用 scatter 实现

    def forward(self, H, edge_index, P_edge=None, deter_edge=None):
        """
        H:           (N, d) 或 (B, N, d)
        edge_index:  (2, E) 或 (B, 2, E)   (src,dst)
        P_edge:      (E,) 或 (B, E) 或 None
        deter_edge:  (E,) 或 (B, E) 或 None
        return:      (N, d) 或 (B, N, d) — 与输入 H 一致
        """
        device = H.device
        batched = _is_batched(H)

        if batched:
            B, N, d = H.shape
            _, _, E = edge_index.shape
            H_flat = H.reshape(B * N, d)
            H_W1 = self.W1(H_flat)
            H_W2 = self.W2(H_flat)
            V     = self.Wv(H_flat)

            # 加批偏移，展平边
            src = edge_index[:, 0, :]  # (B,E)
            dst = edge_index[:, 1, :]  # (B,E)
            bidx = torch.arange(B, device=device).unsqueeze(-1)  # (B,1)
            src_glb = (src + bidx * N).reshape(-1)  # (B*E,)
            dst_glb = (dst + bidx * N).reshape(-1)  # (B*E,)

            P_e = _maybe_flatten_edge_attr(P_edge, B, E, device)  # (B*E,) or None
            deter_e = _maybe_flatten_edge_attr(deter_edge, B, E, device)

            # per-edge 打分
            h_i = H_W1[dst_glb]               # (B*E, d)
            h_j = H_W2[src_glb]               # (B*E, d)
            e_base = h_i + h_j
            if P_e is not None:
                e_base = e_base + self.W3(P_e.view(-1, 1))  # (B*E, d)

            logits = self.W4(e_base)          # (B*E, nh)
            if deter_e is not None:
                logits = logits + deter_e.view(-1, 1)
            logits = self.leaky(logits)

            # segmented softmax（按 dst_glb 分组，逐 head）
            nh = self.nh
            logits_t = logits.permute(1, 0)   # (nh, B*E)
            alphas = []
            dim_size = B * N
            for h in range(nh):
                sh = logits_t[h]  # (B*E,)
                max_per_dst = scatter_max(sh, dst_glb, dim=0, dim_size=dim_size)[0]  # (B*N,)
                max_e = max_per_dst[dst_glb]
                exp_e = (sh - max_e).exp()
                sum_per_dst = scatter_add(exp_e, dst_glb, dim=0, dim_size=dim_size)  # (B*N,)
                alpha_e = exp_e / (sum_per_dst[dst_glb] + 1e-12)
                alphas.append(alpha_e)
            alpha = torch.stack(alphas, dim=1)  # (B*E, nh)
            alpha = self.dropout(alpha)

            # 聚合
            v_e = V[src_glb].view(-1, nh, self.hd)       # (B*E, nh, hd)
            messages = alpha.unsqueeze(-1) * v_e         # (B*E, nh, hd)
            msgs_flat = messages.view(-1, nh * self.hd)  # (B*E, d)
            agg_flat = scatter_add(msgs_flat, dst_glb, dim=0, dim_size=B * N)  # (B*N, d)
            out = agg_flat.view(B, N, d)

            out = self.Wout(out)
            out = self.ln(out + self.res(H))
            return out

        else:
            # 单图路径
            N, d = H.shape
            E = edge_index.shape[1]
            H_W1 = self.W1(H)
            H_W2 = self.W2(H)
            V     = self.Wv(H)

            src = edge_index[0]  # (E,)
            dst = edge_index[1]  # (E,)

            P_e = _maybe_flatten_edge_attr(P_edge, 1, E, device)   # (E,) or None
            deter_e = _maybe_flatten_edge_attr(deter_edge, 1, E, device)

            h_i = H_W1[dst]           # (E,d)
            h_j = H_W2[src]           # (E,d)
            e_base = h_i + h_j
            if P_e is not None:
                e_base = e_base + self.W3(P_e.view(-1, 1))  # (E,d)

            logits = self.W4(e_base)   # (E,nh)
            if deter_e is not None:
                logits = logits + deter_e.view(-1, 1)
            logits = self.leaky(logits)

            nh = self.nh
            logits_t = logits.permute(1, 0)  # (nh,E)
            alphas = []
            for h in range(nh):
                sh = logits_t[h]  # (E,)
                max_per_dst = scatter_max(sh, dst, dim=0, dim_size=N)[0]  # (N,)
                max_e = max_per_dst[dst]
                exp_e = (sh - max_e).exp()
                sum_per_dst = scatter_add(exp_e, dst, dim=0, dim_size=N)  # (N,)
                alpha_e = exp_e / (sum_per_dst[dst] + 1e-12)
                alphas.append(alpha_e)
            alpha = torch.stack(alphas, dim=1)  # (E,nh)
            alpha = self.dropout(alpha)

            v_e = V[src].view(-1, nh, self.hd)        # (E,nh,hd)
            messages = alpha.unsqueeze(-1) * v_e      # (E,nh,hd)
            msgs_flat = messages.view(-1, nh * self.hd)     # (E,d)
            agg = scatter_add(msgs_flat, dst, dim=0, dim_size=N)  # (N,d)

            out = self.Wout(agg)
            out = self.ln(out + self.res(H))
            return out


# ============== CrossSparseGAT（支持单图/Batch） ==============
class CrossSparseGAT(nn.Module):
    """
    dst_feats: (N_dst,d) 或 (B,N_dst,d)
    src_feats: (N_src,d) 或 (B,N_src,d)
    edge_index: (2,E) 或 (B,2,E)  — (src_idx, dst_idx)
    P_edge: (E,) 或 (B,E) 或 None
    deter_edge: (E,) 或 (B,E) 或 None
    returns: 与 dst_feats 同形状
    """
    def __init__(self, args):
        super().__init__()
        d = args.d_model
        self.nh = args.n_heads
        assert d % self.nh == 0, "d_model must be divisible by n_heads"
        self.hd = d // self.nh
        self.d = d

        self.W1 = nn.Linear(d, d, bias=False)  # dst
        self.W2 = nn.Linear(d, d, bias=False)  # src
        self.W3 = nn.Linear(1, d, bias=False)  # edge scalar -> d
        self.W4 = nn.Linear(d, self.nh, bias=False)

        self.Wv = nn.Linear(d, d, bias=False)
        self.Wout = nn.Linear(d, d)
        self.res = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)

        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.1))
        self.use_scatter = True  # 默认使用 scatter 实现

    def forward(self, dst_feats, src_feats, edge_index, P_edge=None, deter_edge=None):
        device = dst_feats.device
        batched = _is_batched(dst_feats)

        if batched:
            B, N_dst, d = dst_feats.shape
            _, N_src, _ = src_feats.shape
            _, _, E = edge_index.shape

            dst_flat = dst_feats.reshape(B * N_dst, d)
            src_flat = src_feats.reshape(B * N_src, d)
            dst_proj = self.W1(dst_flat)
            src_proj = self.W2(src_flat)
            V = self.Wv(src_flat)

            src_idx = edge_index[:, 0, :]  # (B,E)
            dst_idx = edge_index[:, 1, :]  # (B,E)
            bidx = torch.arange(B, device=device).unsqueeze(-1)
            src_glb = (src_idx + bidx * N_src).reshape(-1)  # (B*E,)
            dst_glb = (dst_idx + bidx * N_dst).reshape(-1)  # (B*E,)

            P_e = _maybe_flatten_edge_attr(P_edge, B, E, device)
            deter_e = _maybe_flatten_edge_attr(deter_edge, B, E, device)

            h_i = dst_proj[dst_glb]  # (B*E,d)
            h_j = src_proj[src_glb]  # (B*E,d)
            e_base = h_i + h_j
            if P_e is not None:
                e_base = e_base + self.W3(P_e.view(-1, 1))

            logits = self.W4(e_base)  # (B*E,nh)
            if deter_e is not None:
                logits = logits + deter_e.view(-1, 1)
            logits = self.leaky(logits)

            nh = self.nh
            logits_t = logits.permute(1, 0)  # (nh,B*E)
            alphas = []
            dim_size = B * N_dst
            for h in range(nh):
                sh = logits_t[h]  # (B*E,)
                max_per_dst = scatter_max(sh, dst_glb, dim=0, dim_size=dim_size)[0]
                max_e = max_per_dst[dst_glb]
                exp_e = (sh - max_e).exp()
                sum_per_dst = scatter_add(exp_e, dst_glb, dim=0, dim_size=dim_size)
                alpha_e = exp_e / (sum_per_dst[dst_glb] + 1e-12)
                alphas.append(alpha_e)
            alpha = torch.stack(alphas, dim=1)  # (B*E,nh)
            alpha = self.dropout(alpha)

            v_e = V[src_glb].view(-1, nh, self.hd)  # (B*E,nh,hd)
            messages = alpha.unsqueeze(-1) * v_e
            msgs_flat = messages.view(-1, nh * self.hd)  # (B*E,d)

            agg_flat = scatter_add(msgs_flat, dst_glb, dim=0, dim_size=B * N_dst)  # (B*N_dst,d)
            out = agg_flat.view(B, N_dst, d)

            out = self.Wout(out)
            out = self.ln(out + self.res(dst_feats))
            return out

        else:
            N_dst, d = dst_feats.shape
            N_src, _ = src_feats.shape
            E = edge_index.shape[1]

            dst_proj = self.W1(dst_feats)  # (N_dst,d)
            src_proj = self.W2(src_feats)  # (N_src,d)
            V = self.Wv(src_feats)         # (N_src,d)

            src_idx = edge_index[0]  # (E,)
            dst_idx = edge_index[1]  # (E,)

            P_e = _maybe_flatten_edge_attr(P_edge, 1, E, device)
            deter_e = _maybe_flatten_edge_attr(deter_edge, 1, E, device)

            h_i = dst_proj[dst_idx]   # (E,d)
            h_j = src_proj[src_idx]   # (E,d)
            e_base = h_i + h_j
            if P_e is not None:
                e_base = e_base + self.W3(P_e.view(-1, 1))

            logits = self.W4(e_base)  # (E,nh)
            if deter_e is not None:
                logits = logits + deter_e.view(-1, 1)
            logits = self.leaky(logits)

            nh = self.nh
            logits_t = logits.permute(1, 0)  # (nh,E)
            alphas = []
            for h in range(nh):
                sh = logits_t[h]  # (E,)
                max_per_dst = scatter_max(sh, dst_idx, dim=0, dim_size=N_dst)[0]
                max_e = max_per_dst[dst_idx]
                exp_e = (sh - max_e).exp()
                sum_per_dst = scatter_add(exp_e, dst_idx, dim=0, dim_size=N_dst)
                alpha_e = exp_e / (sum_per_dst[dst_idx] + 1e-12)
                alphas.append(alpha_e)
            alpha = torch.stack(alphas, dim=1)  # (E,nh)
            alpha = self.dropout(alpha)

            v_e = V[src_idx].view(-1, nh, self.hd)  # (E,nh,hd)
            messages = alpha.unsqueeze(-1) * v_e
            msgs_flat = messages.view(-1, nh * self.hd)  # (E,d)

            agg = scatter_add(msgs_flat, dst_idx, dim=0, dim_size=N_dst)  # (N_dst,d)
            out = self.Wout(agg)
            out = self.ln(out + self.res(dst_feats))
            return out


# ============== CoAttLayer（支持单图/Batch） ==============
class CoAttLayer(nn.Module):
    """
    严格 co-attentional block：
      - traj_from_traf: H_traj <- H_traf （edge_index_traj2traf）
      - traf_from_traj: H_traf <- H_traj （edge_index_traf2traj）
    子层后：残差 + LN + FFN
    形状均可为单图或 Batch；与输入保持一致。
    """
    def __init__(self, args):
        super().__init__()
        self.traj_from_traf = CrossSparseGAT(args)
        self.traf_from_traj = CrossSparseGAT(args)

        self.norm_traj1 = nn.LayerNorm(args.d_model)
        self.norm_traj2 = nn.LayerNorm(args.d_model)
        self.norm_traf1 = nn.LayerNorm(args.d_model)
        self.norm_traf2 = nn.LayerNorm(args.d_model)

        self.ffn_traj = nn.Sequential(
            nn.Linear(args.d_model, args.d_model * 4),
            nn.ReLU(),
            nn.Linear(args.d_model * 4, args.d_model),
            nn.Dropout(getattr(args, 'dropout', 0.1)),
        )
        self.ffn_traf = nn.Sequential(
            nn.Linear(args.d_model, args.d_model * 4),
            nn.ReLU(),
            nn.Linear(args.d_model * 4, args.d_model),
            nn.Dropout(getattr(args, 'dropout', 0.1)),
        )

    def forward(self, H_traj, H_traf, edge_index_traf2traj, edge_index_traj2traf,
                P_edge_traf2traj=None, P_edge_traj2traf=None, deter_traf2traj=None, deter_traj2traf=None):

        # traj <- traf
        cross_traj = self.traj_from_traf(
            H_traj, H_traf, edge_index_traj2traf,
            P_edge=P_edge_traj2traf, deter_edge=deter_traj2traf
        )
        H_traj = self.norm_traj1(H_traj + cross_traj)
        H_traj = self.norm_traj2(H_traj + self.ffn_traj(H_traj))

        # traf <- traj
        cross_traf = self.traf_from_traj(
            H_traf, H_traj, edge_index_traf2traj,
            P_edge=P_edge_traf2traj, deter_edge=deter_traf2traj
        )
        H_traf = self.norm_traf1(H_traf + cross_traf)
        H_traf = self.norm_traf2(H_traf + self.ffn_traf(H_traf))

        return H_traj, H_traf
