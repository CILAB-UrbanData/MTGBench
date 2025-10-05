# models/track_model.py
import torch, math, torch.nn as nn, torch.nn.functional as F
from layers.TRACK_attention import StaticSparseGAT, CoAttLayer, CrossSparseGAT

def pairwise_geo_distance(static_feats):
    if static_feats is None: return None
    if static_feats.shape[1] < 2: return None
    coords = static_feats[:, :2]
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dist2 = (diff**2).sum(dim=-1)
    return torch.sqrt(dist2 + 1e-9)

# ---------------- Segment Encoder ----------------
class SegmentEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Linear(args.static_feat_dim, args.d_model)
        self.ln = nn.LayerNorm(args.d_model)
    def forward(self, x):
        return self.ln(F.gelu(self.fc(x)))  # N x d

# ---------------- TrafficTransformer (with temporal emb + X_G injection) ----------------
class TrafficTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.d_model; nh = args.n_heads
        self.input_proj = nn.Linear(2, d)  #2 为输入的flow&speed看情况
        self.weekly_emb = nn.Embedding(7, d)
        self.daily_emb = nn.Embedding(24, d)
        self.pos_emb = nn.Embedding(args.traffic_seq_len + 2, d)
        enc = nn.TransformerEncoderLayer(d, nhead=nh, dim_feedforward=d*2, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=2)
        self.out = nn.Linear(d, d)
    def forward(self, S_hist, X_G=None, weekly_idx=None, daily_idx=None):
        T, N, _ = S_hist.shape
        device = S_hist.device
        x = self.input_proj(S_hist.reshape(T*N, -1)).reshape(T, N, -1)  # T x N x d
        # temporal embeddings
        if weekly_idx is None:
            weekly_idx = torch.zeros(T, dtype=torch.long, device=device)
        if daily_idx is None:
            daily_idx = torch.zeros(T, dtype=torch.long, device=device)
        pos_idx = torch.arange(T, dtype=torch.long, device=device)
        temp = self.weekly_emb(weekly_idx) + self.daily_emb(daily_idx) + self.pos_emb(pos_idx)
        temp_exp = temp.unsqueeze(1).expand(-1, N, -1)  # T x N x d
        x = x + temp_exp
        # inject X_G if exists
        if X_G is not None:
            XG_exp = X_G.unsqueeze(0).expand(T, -1, -1)  # T x N x d
            x = x + XG_exp
        # per-node transformer: N x T x d
        x = x.permute(1,0,2).contiguous()
        out = self.trans(x)  # N x T x d
        last = out[:, -1, :]
        return self.out(last)  # N x d

# ---------------- Trajectory Transformer (cls token, MTP heads) ----------------
class TrajectoryTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.d_model; nh = args.n_heads
        self.node_proj = nn.Linear(d, d)
        self.time_proj = nn.Linear(1, d)
        enc = nn.TransformerEncoderLayer(d, nhead=nh, dim_feedforward=d*2, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=2)
        self.cls = nn.Parameter(torch.randn(1,1,d))
        self.mtp_seg = nn.Linear(d, args.n_nodes)
        self.mtp_time = nn.Linear(d, 1)
    def forward(self, seg_embs, times):
        B,L,_ = seg_embs.shape
        x = self.node_proj(seg_embs) + self.time_proj(times.unsqueeze(-1).float())
        cls = self.cls.expand(B,-1,-1)
        inp = torch.cat([cls, x], dim=1)
        out = self.trans(inp)
        traj_repr = out[:,0,:]
        pos_repr = out[:,1:,:]
        mtp_logits = self.mtp_seg(pos_repr)
        mtp_time = self.mtp_time(pos_repr).squeeze(-1)
        return traj_repr, mtp_logits, mtp_time

# ---------------- Full Model ----------------
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seg = SegmentEncoder(args)
        self.traj_gat = StaticSparseGAT(args)   # 对应论文 (7)/(8) 的 trajectory 内部 GAT（可接受 P_edge_t）
        self.traf_gat = CrossSparseGAT(args)         # traffic view internal / graph GAT（可接受 P_edge）
        self.traf_trans = TrafficTransformer(args)
        self.traj_trans = TrajectoryTransformer(args)
        self.coatt_blocks = nn.ModuleList([CoAttLayer(args) for _ in range(getattr(args, 'n_coatt_layers', 1))])
        self.state_head = nn.Linear(args.d_model, getattr(args, 'state_out_dim', 2))

    # ---------------- 论文公式 (7)/(8)：按时间片 t 使用 P_edge_t（trajectory internal transition-aware GAT） ------------
    def compute_H_traj(self, static, edge_index, P_edge_t=None, deter_edge=None):
        """
        static: N x C
        edge_index: 2 x E
        P_edge_t: E tensor (time-dependent p_{i,j,t}) or None
        deter_edge: optional E tensor
        return: H_traj_t (N x d)
        """
        H0 = self.seg(static)  # N x d
        # 将 static 投影 H0 传入 traj_gat；P_edge_t 会按论文作为 attention 加项
        H_traj_t = self.traj_gat(H0, edge_index, P_edge=P_edge_t, deter_edge=deter_edge)
        return H_traj_t

    # ---------------- traffic view（静态 + 历史 traffic + X_G 注入） ----------------
    def compute_H_traf(self, static, S_hist, edge_index, P_edge=None, weekly_idx=None, daily_idx=None):
        """
        static: N x C
        S_hist: T x N x C_state
        edge_index: 2 x E
        P_edge: E (optional) for graph GAT in traffic branch
        """
        H0 = self.seg(static)  # N x d
        # deter per edge (geometry)
        deter_edge = None
        try:
            dist = pairwise_geo_distance(static)
            if dist is not None:
                sigma = max(1.0, getattr(self.args, 'deter_sigma', 1.0))
                src = edge_index[0]; dst = edge_index[1]
                deter_vals = - (dist[src, dst] ** 2) / (2.0 * (sigma ** 2))
                deter_edge = deter_vals.to(static.device)
        except Exception:
            deter_edge = None

        # traf_gat 用于生成 X_G（graph-level embedding 注入）
        X_G = self.traf_gat(H0, H0, edge_index, P_edge=P_edge, deter_edge=deter_edge)
        H_traf_time = self.traf_trans(S_hist, X_G=X_G, weekly_idx=weekly_idx, daily_idx=daily_idx)
        # 可选融合：论文未明确是否相加 H0；工程中常用 residual：H_traf = H0 + H_traf_time
        H_traf = H0 + H_traf_time
        return H_traf

    # ---------------- co-att: 堆叠多层 CoAttLayer（对应论文 (11)/(12)） ----------------
    def co_attention_exchange(self, H_traj, H_traf, edge_index_traf2traj, edge_index_traj2traf,
                              P_edge_traf2traj=None, P_edge_traj2traf=None, deter_traf2traj=None, deter_traj2traf=None):
        """
        H_traj: N x d
        H_traf: N x d
        edge_index_traf2traj: 2 x E_{t2r} (src in traf, dst in traj)
        edge_index_traj2traf: 2 x E_{r2t} (src in traj, dst in traf)
        P_edge_*: per-edge scalar arrays aligned to edge_index_*
        deter_*: per-edge deterrence aligned to edge_index_*
        """
        # 若未提供 deter_*，可尝试基于 static 计算并传入（也可由调用端传）
        for layer in self.coatt_blocks:
            H_traj, H_traf = layer(H_traj, H_traf,
                                   edge_index_traf2traj, edge_index_traj2traf,
                                   P_edge_traf2traj, P_edge_traj2traf,
                                   deter_traf2traj, deter_traj2traf)
        return H_traj, H_traf

    # ---------------- trajectory encoding（只用 H_traj） ----------------
    def forward_traj(self, H_traj_nodes, traj_nodes_padded, traj_times_padded):
        """
        H_traj_nodes: N x d
        traj_nodes_padded: B x L (indices)
        traj_times_padded: B x L (scalar times)
        返回: traj_repr, mtp_logits, mtp_time
        """
        seg_embs = H_traj_nodes[traj_nodes_padded]  # B x L x d
        return self.traj_trans(seg_embs, traj_times_padded)

    def predict_next_state(self, H):
        return self.state_head(H)