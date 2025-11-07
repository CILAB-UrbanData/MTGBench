# tasks/pretrain.py
"""
Pretrain 脚本（已替换） — 关键点：
- 严格先计算 H_traj = compute_H_traj(static, edge_index)
- 再计算 H_traf = compute_H_traf(static, S_hist, edge_index, P_edge, ...)
- 在两视图都可用后调用 co-attention:
     H_traj, H_traf = model.co_attention_exchange(...)
- Traj 编码使用 co-att 后的 H_traj；matching 使用 co-att 后的 H_traf。
- dataset 需返回 edge_index (2 x E LongTensor) 与 P_edge (E float tensor)
"""

import torch
import numpy as np
import random
import os
from tqdm import tqdm
from utils.metrics import nt_xent_loss, mtp_loss, mtp_time_loss, mask_state_loss, match_nt_xent
from models.TRACK import Model
from data_provider.data_loader import TRACKDataset
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam

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

def run_pretrain(model, dataloader, optimizer, args):
    device = args.device
    model.train()
    model.to(device)
    loss_history = []

    for epoch in range(args.pretrain_epochs):
        running = 0.0
        pbar = tqdm(dataloader, desc=f"Pretrain {epoch+1}/{args.pretrain_epochs}")
        for batch in pbar:
            # required fields from dataset
            full_edge_index  = batch['full_edge_index'].to(device)   # 2 x E (LongTensor)
            P_edge = batch['P_edge_full'].to(device)           # E (FloatTensor)
            static = batch['static'].to(device)           # N x C
            S_hist = batch['S_hist'].to(device)           # T x N x C_state
            edge_index_kmin  = batch['edge_index'].to(device)  # 2 x E_kmin (LongTensor)
            weekly_idx = batch.get('weekly_idx', None)
            daily_idx = batch.get('daily_idx', None)
            if weekly_idx is not None:
                weekly_idx = weekly_idx.to(device)
            if daily_idx is not None:
                daily_idx = daily_idx.to(device)
            traj_pool = batch['trajs']  # list of (nodes, times)

            # ========== 1) compute separate view representations ==========
            # H_traj: strict static-only per paper eq(1)
            H_traj = model.compute_H_traj(static, full_edge_index)  # N x d
            # H_traf: static + traffic history
            H_traf = model.compute_H_traf(static, S_hist, full_edge_index, P_edge=P_edge, weekly_idx=weekly_idx, daily_idx=daily_idx)

            # ========== 2) construct direction-aware edge_index & P_edge for co-att ==========
            # For same node set, we can use reversed edge_index for opposite direction
            edge_index_traf2traj = edge_index_kmin                     # src in traf, dst in traj
            edge_index_traj2traf = torch.stack([edge_index_kmin  [1], edge_index_kmin  [0]], dim=0).to(edge_index_kmin  .device)  # reversed

            # ========== 3) CALL CO-ATTENTION (关键：在两视图都可用时调用) ==========
            # model.co_attention_exchange 应当实现 single-or-multi-layer 的堆叠（模型内部决定层数）
            H_traj, H_traf = model.co_attention_exchange(H_traj, H_traf,
                                                         edge_index_traf2traj=edge_index_traf2traj,
                                                         edge_index_traj2traf=edge_index_traj2traf,
                                                         P_edge_traf2traj=None,
                                                         P_edge_traj2traf=None)

            # ========== 4) downstream: trajectory encoding (use co-attended H_traj) ==========
            B = args.batch_size
            L = args.traj_max_len
            p = float(getattr(args, "aug_dropout", 0.2))

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
            v1 = torch.LongTensor(v1_padded).to(device)
            v2 = torch.LongTensor(v2_padded).to(device)
            N = static.shape[0]
            v1.clamp_(0, N-1)
            v2.clamp_(0, N-1)

            times = torch.as_tensor(times_padded, dtype=torch.float32, device=device)  # (B, L)
            #TODO: V1的维度确定
            # Traj encoding MUST use H_traj (co-attended)
            r1, mtp_logits1, mtp_time1 = model.forward_traj(H_traj, v1, times)  # B x d, B x L x N, B x L
            r2, mtp_logits2, mtp_time2 = model.forward_traj(H_traj, v2, times)

            # CTL
            loss_ctl = nt_xent_loss(r1, r2, args.contrast_temp)

            # MTP losses
            mask = (torch.rand(B, L, device=device) < args.mtp_mask_ratio)
            loss_mtp_seg = mtp_loss(mtp_logits1, v1, mask)
            loss_mtp_time = mtp_time_loss(mtp_time1, times, mask)

            # Traffic mask-state: mask some timesteps and predict last state (simple proxy)
            T = S_hist.shape[0]
            mask_T = (torch.rand(T, device=device) < 0.15)
            S_hist_masked = S_hist.clone()
            S_hist_masked[mask_T] = 0.0
            H_masked = model.compute_H_traf(static, S_hist_masked, full_edge_index , P_edge=P_edge, weekly_idx=weekly_idx, daily_idx=daily_idx)
            pred_next_mask = model.predict_next_state(H_masked)
            true_next_mask = S_hist[:,-1].to(device)
            loss_mask_state = mask_state_loss(pred_next_mask, true_next_mask, mask_T)

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
            loss_match = match_nt_xent(r1, node_avg, args.contrast_temp)

            # total loss
            loss = args.lambda_traj * (loss_ctl + loss_mtp_seg + 0.5 * loss_mtp_time) \
                   + args.lambda_traf * (loss_mask_state + 0.5 * loss_mask_state) \
                   + args.lambda_match * loss_match

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'ctl': loss_ctl.item()})

        epoch_loss = running / max(1, len(dataloader))
        loss_history.append(epoch_loss)

    # save checkpoint
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/pretrained.pth')
    np.save('checkpoints/loss_pretrain.npy', np.array(loss_history))
    print("Pretrain done: checkpoints/pretrained.pth, loss_pretrain.npy saved")

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--static_feat_dim', type=int, default=3)
    argparser.add_argument('--d_model', type=int, default=32, help='Dimension of the model')
    argparser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    argparser.add_argument('--traffic_seq_len', type=int, default=7)
    argparser.add_argument('--n_nodes', type=int, default=26659)
    argparser.add_argument('--pretrain_epochs', type=int, default=100, help='Number of pretraining epochs')
    argparser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    argparser.add_argument('--gpu', type=int, default=0, help='gpu')
    argparser.add_argument('--batch_size', type=int, default=4, help='Batch size for pretraining')
    argparser.add_argument('--traj_max_len', type=int, default=30, help='Max length of trajectory for pretraining')
    argparser.add_argument('--aug_dropout', type=float, default=0.2, help='Node dropout ratio for trajectory augmentation')
    argparser.add_argument('--contrast_temp', type=float, default=0.1, help='Temperature for contrastive loss')
    argparser.add_argument('--mtp_mask_ratio', type=float, default=0.15, help='Mask ratio for MTP task')
    argparser.add_argument('--lambda_traj', type=float, default=1.0, help='Weight for trajectory losses')
    argparser.add_argument('--lambda_traf', type=float, default=1.0, help='Weight for traffic view losses')
    argparser.add_argument('--lambda_match', type=float, default=1.0, help='Weight for matching loss')
    args = argparser.parse_args()


    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')
    dataset = TRACKDataset("data/sf_data/raw")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True)
    N = int(dataset.static.shape[0])  
    has_unk = (len(getattr(dataset, "idx2seg", [])) > 0 and dataset.idx2seg[-1] == "UNK")
    N_graph = N - 1 if has_unk else N   
    args.n_nodes = N    
    print(f"[pretrain] N_total={N}, N_graph={N_graph}, has_unk={has_unk}")

    model = Model(args)  # Initialize your model with appropriate args
    optimizer = Adam(model.parameters(), lr=0.001)

    run_pretrain(model, dataloader, optimizer, args)