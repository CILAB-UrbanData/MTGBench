# models/track_model.py
import torch, math
import torch.nn as nn
import torch.nn.functional as F
import random
import os
from layers.TRACK_attention import StaticSparseGAT, CoAttLayer, CrossSparseGAT

def pairwise_geo_distance(static_feats):
    if static_feats is None: return None
    if static_feats.shape[1] < 2: return None
    coords = static_feats[:, :2]
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)
    dist2 = (diff**2).sum(dim=-1)
    return torch.sqrt(dist2 + 1e-9)

# padding helper
def pad_trajs(seqs, L, pad_val_getter):
    out = []
    for seq in seqs:
        if len(seq) >= L:
            out.append(seq[:L])
        else:
            pad_val = pad_val_getter(seq)  # 通常用 seq[-1]
            out.append(seq + [pad_val] * (L - len(seq)))
    return out

# ---------------- Segment Encoder ----------------
class SegmentEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc = nn.Linear(args.static_feat_dim, args.d_model)
        self.ln = nn.LayerNorm(args.d_model)
    def forward(self, x):
        return self.ln(F.gelu(self.fc(x)))  # B x N x d

# ---------------- TrafficTransformer (with temporal emb + X_G injection) ----------------
class TrafficTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.d_model; nh = args.n_heads
        self.input_proj = nn.LazyLinear(args.d_model)    #2 为输入的flow&speed看情况
        self.weekly_emb = nn.Embedding(7, d)
        self.daily_emb = nn.Embedding(24, d)
        self.pos_emb = nn.Embedding(args.traffic_seq_len + 2, d)
        enc = nn.TransformerEncoderLayer(d, nhead=nh, dim_feedforward=d*2, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=3)
        self.out = nn.Linear(d, d)
    def forward(self, S_hist, X_G=None, weekly_idx=None, daily_idx=None):
        B, T, N, _ = S_hist.shape
        device = S_hist.device
        x = self.input_proj(S_hist.reshape(B*T*N, -1)).reshape(B, T, N, -1)  # B x T x N x d
        # temporal embeddings
        if weekly_idx is None:
            weekly_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        if daily_idx is None:
            daily_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        #pos = torch.arange(T, dtype=torch.long, device=device).unsqueeze(0)  # 1 x T
        temp = self.weekly_emb(weekly_idx) + self.daily_emb(daily_idx) #+ self.pos_emb(pos)  # B x T x d
        x = x + temp.unsqueeze(2).expand(B, T, N, -1)  # B x T x N x d
        # inject X_G if exists
        if X_G is not None:
            x = x + X_G.unsqueeze(1).expand(B, T, N, -1)  # B x T x N x d
        # per-node transformer: N x T x d
        x_seq = x.permute(0, 2, 1, 3).contiguous().view(B*N, T, -1)
        
        #TODO:把chunk继承一下
        #B*N过大会导致cuda内核报错，因此要分块加载 2025/11/19
        chunk = 4096  # 可以试 2048/4096/8192，看显存和速度
        outs = []
        for s in range(0, x_seq.size(0), chunk):
            e = min(s + chunk, x_seq.size(0))
            outs.append(self.trans(x_seq[s:e]))  # 每次只对一部分节点做注意力
        out = torch.cat(outs, dim=0)      
        
        # out = self.trans(x_seq)  # N x T x d
        last = out[:, -1, :].view(B, N, -1)              # 取最后时刻的输出 B x N x d
        return self.out(last)  # B x N x d

# ---------------- Trajectory Transformer (cls token, MTP heads) ----------------
class TrajectoryTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        d = args.d_model; nh = args.n_heads
        self.node_proj = nn.Linear(d, d)
        self.time_proj = nn.Linear(1, d)
        enc = nn.TransformerEncoderLayer(d, nhead=nh, dim_feedforward=d*2, batch_first=True)
        self.trans = nn.TransformerEncoder(enc, num_layers=6)
        self.cls = nn.Parameter(torch.randn(1,1,d))
        self.mtp_seg = nn.Linear(d, args.NumofRoads)
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
        return traj_repr, mtp_logits, mtp_time  #轨迹整体表征 (CLS), 每个位置的节点预测 (分类), 多目标时间预测

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
        self.coatt_blocks = nn.ModuleList([CoAttLayer(args) for _ in range(getattr(args, 'n_coatt_layers', 2))])
        self.state_head = nn.Linear(args.d_model, getattr(args, 'pre_steps', 1))
        if self.args.load_pretrained and os.path.exists(f'checkpoints/TRACK_pretrain_test_TRACK_{args.data}_lr_{args.learning_rate}_ba_{args.batch_size}_epo_{args.train_epochs}_itr_0/checkpoint.pth'):
            self.load_state_dict(torch.load(f'checkpoints/TRACK_pretrain_test_TRACK_{args.data}_lr_{args.learning_rate}_ba_{args.batch_size}_epo_{args.train_epochs}_itr_0/checkpoint.pth', 
                                            map_location=self.args.device))
            print("Loaded pretrained.pth")

    # --------- 工具：把 edge_index 复制 B 份并做偏移（+b*N） ----------
    def _tile_edge_index(self, edge_index: torch.Tensor, B: int, N: int) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=next(self.parameters()).device)
        ei = edge_index.long().to(next(self.parameters()).device)        # 2 x E
        B, N = int(B), int(N)
        E   = ei.size(1)
        eiB = ei.unsqueeze(0).expand(B, -1, -1).contiguous()            # B x 2 x E
        off = (torch.arange(B, device=eiB.device).view(B,1,1) * N).long()
        eiB[:,0,:] = eiB[:,0,:] + off.squeeze(-1)                       # src 偏移
        eiB[:,1,:] = eiB[:,1,:] + off.squeeze(-1)                       # dst 偏移
        return eiB.permute(1,0,2).reshape(2, B*E)                        # 2 x (B*E)

    def _edge_geo_det(self, static_batch: torch.Tensor, eiB: torch.Tensor) -> torch.Tensor | None:
        """
        static_batch: B x N x C  (first two dims of C are x,y)
        eiB: 2 x (B*E)           (tiled edge_index with +b*N offsets)
        return: (B*E,) tensor of -||x_i - x_j||^2 / (2 sigma^2), or None if no coords.
        """
        if static_batch.size(-1) < 2:
            return None
        B, N, C = static_batch.shape
        device = static_batch.device
        xy = static_batch[..., :2].contiguous().view(B * N, 2).to(device)  # (B*N) x 2
        src_xy = xy.index_select(0, eiB[0])
        dst_xy = xy.index_select(0, eiB[1])
        diff = src_xy - dst_xy
        d2 = (diff * diff).sum(dim=-1)                                     # (B*E,)
        sigma = float(getattr(self.args, "deter_sigma", 1.0))
        if sigma <= 0:  # disable if misconfigured
            return None
        return -0.5 * (d2 / (sigma * sigma))                               # (B*E,)

    # ---------------- 论文公式 (7)/(8)：按时间片 t 使用 P_edge_t（trajectory internal transition-aware GAT） ------------
    def _compute_H_traj(self, static, edge_index, P_edge_t=None, deter_edge=None):
        B, N, C = static.shape
        H0 = self.seg(static.reshape(B * N, C))
        eiB = self._tile_edge_index(edge_index, B, N)                      # 2 x (B*E)
        P_edge_flat = None if (P_edge_t is None or P_edge_t.numel() == 0) \
                        else P_edge_t.reshape(-1).to(H0.device)
        # geometric deterrence from coords
        det_flat = self._edge_geo_det(static, eiN := eiB) if deter_edge is None else \
                (deter_edge.reshape(-1).to(H0.device))
        H = self.traj_gat(H0, eiB, P_edge_flat, det_flat)
        return H.view(B, N, -1)

    # ---------------- traffic view（静态 + 历史 traffic + X_G 注入） ----------------
    def _compute_H_traf(self, static, S_hist, edge_index, P_edge=None, weekly_idx=None, daily_idx=None):
        B, N, C = static.shape
        H0  = self.seg(static.reshape(B * N, C))
        eiB = self._tile_edge_index(edge_index, B, N)
        P_edge_flat = None if (P_edge is None or P_edge.numel() == 0) \
                        else P_edge.reshape(-1).to(H0.device)
        det_flat = self._edge_geo_det(static, eiB)
        # graph-side GAT to produce X_G
        XG = self.traf_gat(H0, H0, eiB, P_edge=P_edge_flat, deter_edge=det_flat).view(B, N, -1)
        # temporal encoder (batch)
        H_time = self.traf_trans(S_hist, X_G=XG, weekly_idx=weekly_idx, daily_idx=daily_idx)  # B x N x d
        return H0.view(B, N, -1) +  H_time

    # ---------------- co-att: 堆叠多层 CoAttLayer（对应论文 (11)/(12)） ----------------
    def _co_attention_exchange(self, H_traj, H_traf, edge_index_traf2traj, edge_index_traj2traf,
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
        B, N, d = H_traj.shape
        Ht = H_traj.view(B*N, d)
        Hf = H_traf.view(B*N, d)
        ei_t2r = self._tile_edge_index(edge_index_traf2traj, B, N)
        ei_r2t = self._tile_edge_index(edge_index_traj2traf, B, N)
        p_t2r = None if (P_edge_traf2traj is None) else P_edge_traf2traj.reshape(-1)
        p_r2t = None if (P_edge_traj2traf is None) else P_edge_traj2traf.reshape(-1)
        for layer in self.coatt_blocks:
            Ht, Hf = layer(Ht, Hf, ei_t2r, ei_r2t, p_t2r, p_r2t, deter_traf2traj, deter_traj2traf)
        return Ht.view(B, N, d), Hf.view(B, N, d)

    # ---------------- trajectory encoding（只用 H_traj） ----------------
    def _forward_traj(self, H_traj_nodes, traj_nodes_padded, traj_times_padded):
        """
        H_traj_nodes: N x d
        traj_nodes_padded: B x L (indices)
        traj_times_padded: B x L (scalar times)
        返回: traj_repr, mtp_logits, mtp_time
        """
        B, N, d = H_traj_nodes.shape
        B2, L = traj_nodes_padded.shape
        assert B == B2, "batch size mismatch"
        bidx = torch.arange(B, device=H_traj_nodes.device).unsqueeze(1).expand(B, L)
        seg_embs = H_traj_nodes[bidx, traj_nodes_padded, :]  # B x L x d
        return self.traj_trans(seg_embs, traj_times_padded)

    def predict_next_state(self, H):
        return self.state_head(H)
    
    # 对于pretrain阶段的forward封装
    def forward_pretrain(self, batch):
        """
        完整的 pretrain forward 流程
        """
        full_edge_index  = batch['full_edge_index'].to(self.args.device)   # 2 x E (LongTensor)
        P_edge = batch['P_edge_full'].to(self.args.device)           # E (FloatTensor)
        static = batch['static'].to(self.args.device)           # N x C
        S_hist = batch['S_hist'].to(self.args.device)           # T x N x C_state
        edge_index_kmin = batch['edge_index'].to(self.args.device)  # 2 x E_kmin (LongTensor)
        weekly_idx = batch.get('weekly_idx', None)
        daily_idx = batch.get('daily_idx', None)
        if weekly_idx is not None:
            weekly_idx = weekly_idx.to(self.args.device)
        if daily_idx is not None:
            daily_idx = daily_idx.to(self.args.device)
        traj_pool = batch['trajs']  # list of (nodes, times)   

        # ========== 1) compute separate view representations ==========
        # H_traj: strict static-only per paper eq(1)
        H_traj = self._compute_H_traj(static, full_edge_index)  # N x d
        # H_traf: static + traffic history    最后几个数据是真值
        H_traf = self._compute_H_traf(static, S_hist[:,:-self.args.pre_steps], 
                                      full_edge_index, P_edge=P_edge, weekly_idx=weekly_idx[:,:-self.args.pre_steps], daily_idx=daily_idx[:,:-self.args.pre_steps])     

        # ========== 2) construct direction-aware edge_index & P_edge for co-att ==========
        # For same node set, we can use reversed edge_index for opposite direction
        edge_index_traf2traj = edge_index_kmin                     # src in traf, dst in traj
        edge_index_traj2traf = torch.stack([edge_index_kmin[1], edge_index_kmin[0]], dim=0).to(edge_index_kmin.device)  # reversed 

        # ========== 3) CALL CO-ATTENTION (关键：在两视图都可用时调用) ==========
        # model.co_attention_exchange 应当实现 single-or-multi-layer 的堆叠（模型内部决定层数）
        H_traj, H_traf = self._co_attention_exchange(H_traj, H_traf,
                                                        edge_index_traf2traj=edge_index_traf2traj,
                                                        edge_index_traj2traf=edge_index_traj2traf,
                                                        P_edge_traf2traj=None,
                                                        P_edge_traj2traf=None)  
         
        # ========== 4) downstream: trajectory encoding (use co-attended H_traj) ==========
        B = self.args.batch_size
        L = self.args.traj_max_len
        p = float(getattr(self.args, "aug_dropout", 0.2))     

        # 兜底：若 traj_pool 不是长度 B，则尽量对齐（常见是 B=1 的情况）
        if not isinstance(traj_pool, list):
            trajs_lists = [[] for _ in range(B)]
        elif len(traj_pool) == B and (len(traj_pool) == 0 or isinstance(traj_pool[0], list)):
            trajs_lists = traj_pool
        elif len(traj_pool) == 1 and isinstance(traj_pool[0], list) and B > 1:
            # 只有一个候选列表，但 batch 想要 B 条，就重复使用该列表
            trajs_lists = [traj_pool[0] for _ in range(B)]
        else:
            # 最差兜底：把所有候选合并后，均分/重复
            flat = [tb for tlist in traj_pool for tb in (tlist if isinstance(tlist, list) else [])]
            if len(flat) == 0:
                trajs_lists = [[] for _ in range(B)]
            else:
                trajs_lists = [flat for _ in range(B)]

        # traj_pool: [[([node_seq],[time_bin_seq]), ...]]
        # 采样 B 条轨迹（全局随机）
        # 每个样本各抽一条轨迹 (nodes, bins)
        picked = []
        for b in range(B):
            cand = trajs_lists[b]
            if isinstance(cand, list) and len(cand) > 0:
                picked.append(random.choice(cand))
            else:
                picked.append(([], []))  # 空则占位
        
        nodes_list = [seq if isinstance(seq, (list, tuple)) else list(seq) for (seq, _) in picked]
        bins_list  = [ts  if isinstance(ts,  (list, tuple)) else list(ts)  for (_, ts) in picked]
        # 防止空轨迹：用 [0] 占位（后续会 clamp 到 [0, N-1]）
        nodes_list = [seq if len(seq) > 0 else [0] for seq in nodes_list]
        bins_list  = [ts  if len(ts)  > 0 else [0] for ts  in bins_list]

        # 视图1：按概率丢弃（至少保留一个）
        v1_nodes = [[n for n in seq if random.random() > p] or [seq[0]] for seq in nodes_list]

        # 视图2：按概率扰动为邻近 id（mod 到合法范围）
        N = static.shape[0]  # 含 UNK
        # 视图2：按概率扰动为邻近 id
        v2_nodes = [[((n + random.randint(-2, 2)) % N) if random.random() < p else n for n in seq]
                    for seq in nodes_list]

        v1_padded   = pad_trajs(v1_nodes, L, lambda s: s[-1] if len(s) > 0 else 0)
        v2_padded   = pad_trajs(v2_nodes, L, lambda s: s[-1] if len(s) > 0 else 0)
        times_padded = pad_trajs(bins_list, L, lambda s: s[-1] if len(s) > 0 else 0)
        v1 = torch.LongTensor(v1_padded).to(self.args.device)
        v2 = torch.LongTensor(v2_padded).to(self.args.device)
        N = static.shape[0]
        v1.clamp_(0, N-1)
        v2.clamp_(0, N-1)

        times = torch.as_tensor(times_padded, dtype=torch.float32, device=self.args.device)  # (B, L)
        # Traj encoding MUST use H_traj (co-attended)
        r1, mtp_logits1, mtp_time1 = self._forward_traj(H_traj, v1, times)  # B x d, B x L x N, B x L
        r2, mtp_logits2, mtp_time2 = self._forward_traj(H_traj, v2, times)

        # MTP losses
        mask = (torch.rand(B, L, device=self.args.device) < self.args.mtp_mask_ratio)

        # Matching: use co-attended H_traf aggregated over trajectory nodes
        if H_traf.dim() == 2:  # (N, d)
            node_embs = H_traf[v1]  # (B, L, d)
        elif H_traf.dim() == 3:  # (B, N, d)
            B = H_traf.size(0)
            bidx = torch.arange(B, device=H_traf.device).unsqueeze(-1).expand_as(v1)
            node_embs = H_traf[bidx, v1, :]  # (B, L, d)
        else:
            raise ValueError(f"Unexpected H_traf shape: {H_traf.shape}")
        node_avg = node_embs.mean(dim=1)  # (B, d)

        T = S_hist.shape[0]
        mask_T = (torch.rand(T, device=self.args.device) < 0.15)
        S_hist_masked = S_hist[:,:-self.args.pre_steps].clone()
        S_hist_masked[mask_T] = 0.0
        H_masked = self._compute_H_traf(static, S_hist_masked, full_edge_index , P_edge=P_edge, weekly_idx=weekly_idx[:,-self.args.pre_steps:], daily_idx=daily_idx[:,-self.args.pre_steps:])
        pred_next_mask = self.predict_next_state(H_masked)
        true_next_mask = S_hist[:,-self.args.pre_steps].reshape(B,S_hist_masked.shape[2],-1).to(self.args.device)

        return (r1, mtp_logits1, mtp_time1), (r2, mtp_logits2, mtp_time2), mask, v1, times, node_avg, pred_next_mask, true_next_mask, mask_T

    def forward(self, batch):
        """
        标准的预测任务 forward 流程
        """
        full_edge_index  = batch['full_edge_index'].to(self.args.device)   # 2 x E (LongTensor)
        P_edge = batch['P_edge_full'].to(self.args.device)           # E (FloatTensor)
        static = batch['static'].to(self.args.device)           # N x C
        S_hist = batch['S_hist'].to(self.args.device)           # T x N x C_state
        edge_index_kmin = batch['edge_index'].to(self.args.device)  # 2 x E_kmin (LongTensor)
        weekly_idx = batch.get('weekly_idx', None)
        daily_idx = batch.get('daily_idx', None)
        if weekly_idx is not None:
            weekly_idx = weekly_idx.to(self.args.device)
        if daily_idx is not None:
            daily_idx = daily_idx.to(self.args.device)
        traj_pool = batch['trajs']  # list of (nodes, times)   

        # ========== 1) compute separate view representations ==========
        # H_traj: strict static-only per paper eq(1)
        H_traj = self._compute_H_traj(static, full_edge_index)  # N x d
        # H_traf: static + traffic history    最后几个数据是真值
        H_traf = self._compute_H_traf(static, S_hist[:,:-self.args.pre_steps], 
                                      full_edge_index, P_edge=P_edge, weekly_idx=weekly_idx[:,:-self.args.pre_steps], daily_idx=daily_idx[:,:-self.args.pre_steps])     

        # ========== 2) construct direction-aware edge_index & P_edge for co-att ==========
        # For same node set, we can use reversed edge_index for opposite direction
        edge_index_traf2traj = edge_index_kmin                     # src in traf, dst in traj
        edge_index_traj2traf = torch.stack([edge_index_kmin[1], edge_index_kmin[0]], dim=0).to(edge_index_kmin.device)  # reversed 

        # ========== 3) CALL CO-ATTENTION (关键：在两视图都可用时调用) ==========
        # model.co_attention_exchange 应当实现 single-or-multi-layer 的堆叠（模型内部决定层数）
        H_traj, H_traf = self._co_attention_exchange(H_traj, H_traf,
                                                        edge_index_traf2traj=edge_index_traf2traj,
                                                        edge_index_traj2traf=edge_index_traj2traf,
                                                        P_edge_traf2traj=None,
                                                        P_edge_traj2traf=None)  
        pred_next_mask = self.predict_next_state(H_traf)

        return pred_next_mask
        