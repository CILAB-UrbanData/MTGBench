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
from data_provider.track_data import SegmentRoadDataset, batch_collate_fn
import argparse
from torch.utils.data import DataLoader
from torch.optim import Adam

# padding helper
def pad_trajs(nodes_list, L):
    res = []
    for seq in nodes_list:
        if len(seq) >= L:
            res.append(seq[:L])
        else:
            res.append(seq + [seq[-1]] * (L - len(seq)))
    return res

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
            sampled_nodes = [random.choice(traj_pool)[0] for _ in range(B)]
            # two augmented views for contrastive learning
            v1 = [ [n for n in seq if random.random() > args.aug_dropout] or [seq[0]] for seq in sampled_nodes ]
            v2 = [ [ (n + random.randint(-2,2)) % args.n_nodes if random.random() < args.aug_dropout else n for n in seq ] for seq in sampled_nodes ]
            v1 = pad_trajs(v1, L)
            v2 = pad_trajs(v2, L)
            v1 = torch.LongTensor(v1).to(device)
            v2 = torch.LongTensor(v2).to(device)
            times = torch.zeros((B, L), device=device)

            # Traj encoding MUST use H_traj (co-attended)
            r1, mtp_logits1, mtp_time1 = model.forward_traj(H_traj, v1, times)
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
            true_next_mask = S_hist[-1].to(device)
            loss_mask_state = mask_state_loss(pred_next_mask, true_next_mask, mask_T)

            # Matching: use co-attended H_traf aggregated over trajectory nodes
            node_embs = H_traf[v1]  # B x L x d
            node_avg = node_embs.mean(dim=1)
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
    argparser.add_argument('--pretrain_epochs', type=int, default=2, help='Number of pretraining epochs')
    argparser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    argparser.add_argument('--gpu', type=int, default=0, help='gpu')
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

    model = Model(args)  # Initialize your model with appropriate args
    optimizer = Adam(model.parameters(), lr=0.001)
    dataset = SegmentRoadDataset("data/sf_data/raw")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=batch_collate_fn)
    run_pretrain(model, dataloader, optimizer, args)


# #!/usr/bin/env python3
# # tasks/pretrain.py
# """
# Pretraining script for TRACK-style model (batch-aware).
# Assumes:
#   - dataset/segment_road_dataset.py exports SegmentRoadDataset and batch_collate_fn
#   - models/track_model_batch.py exports TrackMiniPaper (batch-aware)
# This script performs:
#   - compute H_traj_batch using full_edge_index + P_edge_full (trajectory static GAT)
#   - compute H_traf_batch using traffic history + X_G injection (traffic transformer)
#   - co-attention exchange (kmin edges)
#   - trajectory encoding for sampled trajectories (two augmented views)
#   - contrastive NT-Xent loss between traj repr and traffic repr (per-sample positive)
#   - optional MTP segmentation prediction loss (cross-entropy)
# """

# import os
# import math
# import random
# import argparse
# from tqdm import tqdm
# from collections import defaultdict
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader

# # local imports - ensure paths are correct in your project
# from dataset.segment_road_dataset import SegmentRoadDataset, batch_collate_fn
# from models.track_model_batch import TrackMiniPaper

# # ---------------- utilities ----------------
# def pad_trajs(list_of_trajs, L, pad_value=0):
#     """
#     list_of_trajs: list of lists of node indices (int)
#     returns: np.array shape (B, L) padded with pad_value, and mask B x L (1 for real tokens)
#     """
#     B = len(list_of_trajs)
#     out = np.full((B, L), pad_value, dtype=np.int64)
#     mask = np.zeros((B, L), dtype=np.bool_)
#     for i, seq in enumerate(list_of_trajs):
#         ln = min(len(seq), L)
#         if ln > 0:
#             out[i, :ln] = seq[:ln]
#             mask[i, :ln] = True
#     return out, mask

# def sample_one_traj_from_pool(traj_pool):
#     """
#     traj_pool: list of (nodes_list, bins_list) for a given sample/time_bin
#     returns: nodes_list (list of int), bins_list
#     """
#     if not traj_pool:
#         return [], []
#     return random.choice(traj_pool)

# def augment_traj_dropout(nodes, dropout_p=0.2):
#     """
#     Simple augmentation: randomly drop some nodes (with min 1)
#     """
#     if len(nodes) == 0:
#         return nodes
#     out = [n for n in nodes if random.random() > dropout_p]
#     if len(out) == 0:
#         out = [nodes[0]]
#     return out

# def augment_traj_local_shift(nodes, shift_range=2, vocab_size=None):
#     """
#     Length-preserving local shift: with some prob, shift a node id by +/- small int (wrap-around)
#     Only used if you want synthetic perturbation of node ids (not ideal for real seg ids).
#     We'll avoid warp if vocab_size is None.
#     """
#     if len(nodes) == 0:
#         return nodes
#     out = []
#     for n in nodes:
#         if random.random() < 0.1 and vocab_size:
#             delta = random.randint(-shift_range, shift_range)
#             out.append((n + delta) % vocab_size)
#         else:
#             out.append(n)
#     return out

# def nt_xent_loss(z_a, z_b, temperature=0.1):
#     """
#     Compute symmetric NT-Xent between two sets of representations and their positives.
#     Here we assume z_a and z_b are two views for the same B items, and positives are (a_i, b_i).
#     We'll compute loss = -log( exp(sim(a_i,b_i)/tau) / sum_j exp(sim(a_i, b_j)/tau) )
#     and symmetric average.
#     Inputs:
#        z_a: B x d
#        z_b: B x d
#     returns scalar loss
#     """
#     device = z_a.device
#     B = z_a.size(0)
#     z_a = F.normalize(z_a, dim=1)
#     z_b = F.normalize(z_b, dim=1)
#     # similarity matrix
#     sims = torch.mm(z_a, z_b.t())  # B x B
#     sims_div = sims / temperature
#     # positive logits are diagonal
#     labels = torch.arange(B, device=device)
#     loss_a = F.cross_entropy(sims_div, labels)  # treats row i: similarities to all b_j, target is j=i
#     loss_b = F.cross_entropy(sims_div.t(), labels)
#     return 0.5 * (loss_a + loss_b)

# def contrast_traj_vs_traf(traj_repr, traf_repr, temperature=0.1):
#     """
#     traj_repr: B x d  (from forward_traj view)
#     traf_repr: B x d  (aggregated traffic repr per sample)
#     We treat positive pairs (traj_i, traf_i). Negatives are all other traf_j.
#     Use NT-Xent symmetry (traj->traf and traf->traj) for stability.
#     """
#     return nt_xent_loss(traj_repr, traf_repr, temperature)

# # ---------------- main pretrain routine ----------------
# def train(args):
#     device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
#     print("Using device:", device)

#     # dataset & dataloader
#     ds = SegmentRoadDataset(data_root=args.data_root,
#                             traj_file=args.traj_file,
#                             edge_list_file=args.edge_shp if args.edge_shp else None,
#                             static_file=args.static_file if args.static_file else None,
#                             traffic_ts_file=args.traffic_ts_file if args.traffic_ts_file else None,
#                             num_time_bins=args.num_time_bins,
#                             T_hist=args.traffic_seq_len,
#                             K_min=args.K_min,
#                             cache_dir=args.cache_dir,
#                             force_recompute=args.force_recompute,
#                             khop_fallback=args.khop_fallback)
#     dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=batch_collate_fn, num_workers=args.num_workers)

#     # model
#     model = TrackMiniPaper(args).to(device)
#     optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     # auxiliary losses
#     ce_loss = nn.CrossEntropyLoss(ignore_index=-1)  # we'll map pad positions to -1 if needed

#     global_step = 0
#     model.train()
#     for epoch in range(args.epochs):
#         pbar = tqdm(dl, desc=f"Pretrain epoch {epoch}", leave=False)
#         for batch in pbar:
#             # move shared items to device
#             full_edge_index = batch['full_edge_index'].to(device)  # 2 x E_full
#             edge_index_kmin = batch['edge_index'].to(device)       # 2 x E_k
#             P_edge_full_batch = batch['P_edge_full'].to(device)   # B x E_full
#             static_batch = batch['static'].to(device)             # B x N x C_static
#             S_hist_batch = batch['S_hist'].to(device)             # B x T_hist x N x C_state
#             weekly_idx = batch.get('weekly_idx', None)
#             daily_idx = batch.get('daily_idx', None)
#             if weekly_idx is not None:
#                 weekly_idx = weekly_idx.to(device)
#             if daily_idx is not None:
#                 daily_idx = daily_idx.to(device)

#             B = static_batch.size(0)

#             # ---------------- compute node embeddings per view ----------------
#             # 1) compute H_traj_batch: using full edges + P_edge_full_batch
#             H_traj_batch = model.compute_H_traj_batch(static_batch, full_edge_index, P_edge_full_batch, deter_edge_batch=None)
#             # 2) compute H_traf_batch: using traffic history + X_G injection (uses edge_index_kmin internally)
#             H_traf_batch = model.compute_H_traf_batch(static_batch, S_hist_batch, edge_index_kmin,
#                                                       P_edge_batch=None, weekly_idx=weekly_idx, daily_idx=daily_idx)

#             # 3) co-attention exchange (kmin edges)
#             edge_index_traf2traj = edge_index_kmin
#             edge_index_traj2traf = torch.stack([edge_index_kmin[1], edge_index_kmin[0]], dim=0).to(device)
#             H_traj_batch, H_traf_batch = model.co_attention_exchange_batch(H_traj_batch, H_traf_batch,
#                                                                            edge_index_traf2traj=edge_index_traf2traj,
#                                                                            edge_index_traj2traf=edge_index_traj2traf,
#                                                                            P_edge_traf2traj=None, P_edge_traj2traf=None,
#                                                                            deter_traf2traj=None, deter_traj2traf=None)
#             # ---------------- sample & build trajectory mini-batch ----------------
#             # For each sample in batch, pick one trajectory from batch['trajs'][i] (which is a list of trajs for that sample)
#             traj_pool_batch = batch['trajs']  # list len B, each element list of traj tuples (nodes, bins)
#             trajs_v1 = []
#             trajs_v2 = []
#             orig_trajs = []
#             for i in range(B):
#                 pool = traj_pool_batch[i]
#                 nodes, bins = sample_one_traj_from_pool(pool)  # nodes are already segment idx
#                 if len(nodes) == 0:
#                     # fallback: choose random dummy sequence (e.g., first node repeated)
#                     nodes = [0]
#                 orig_trajs.append(nodes)
#                 # augmentation view 1: dropout
#                 v1 = augment_traj_dropout(nodes, dropout_p=args.aug_dropout)
#                 # augmentation view 2: local shift
#                 v2 = augment_traj_local_shift(nodes, shift_range=2, vocab_size=args.n_nodes)
#                 trajs_v1.append(v1)
#                 trajs_v2.append(v2)

#             L = args.traj_max_len
#             v1_arr, v1_mask = pad_trajs(trajs_v1, L, pad_value=0)
#             v2_arr, v2_mask = pad_trajs(trajs_v2, L, pad_value=0)
#             orig_arr, orig_mask = pad_trajs(orig_trajs, L, pad_value=0)

#             v1_ts = np.zeros_like(v1_arr, dtype=np.float32)  # times currently unused by forward_traj (can be zeros)
#             v2_ts = np.zeros_like(v2_arr, dtype=np.float32)
#             orig_ts = np.zeros_like(orig_arr, dtype=np.float32)

#             v1_t = torch.LongTensor(v1_arr).to(device)       # B x L
#             v2_t = torch.LongTensor(v2_arr).to(device)
#             orig_t = torch.LongTensor(orig_arr).to(device)
#             v1_times = torch.FloatTensor(v1_ts).to(device)
#             v2_times = torch.FloatTensor(v2_ts).to(device)
#             orig_times = torch.FloatTensor(orig_ts).to(device)

#             # ---------------- encode trajectories ----------------
#             # forward_traj supports batch H_traj_batch input and B x L node lists
#             r1, mtp_logits1, mtp_time1 = model.forward_traj(H_traj_batch, v1_t, v1_times)
#             r2, mtp_logits2, mtp_time2 = model.forward_traj(H_traj_batch, v2_t, v2_times)
#             # optionally encode orig for supervision
#             r_orig, mtp_logits_orig, mtp_time_orig = model.forward_traj(H_traj_batch, orig_t, orig_times)

#             # ---------------- build traffic representation for each sample (positive) ----------------
#             # We aggregate H_traf_batch[b, traj_nodes] mean as the traffic representation aligned with the trajectory
#             B, N, d = H_traf_batch.shape
#             traf_reprs = []
#             for i in range(B):
#                 nodes = orig_trajs[i][:L] if len(orig_trajs[i]) > 0 else [0]
#                 nodes_idx = torch.LongTensor(nodes).to(device)
#                 # avoid empty
#                 if nodes_idx.numel() == 0:
#                     traf_reprs.append(H_traf_batch[i].mean(dim=0))
#                 else:
#                     # clamp indices < N
#                     nodes_idx = nodes_idx.clamp(0, N-1)
#                     tr = H_traf_batch[i, nodes_idx, :].mean(dim=0)
#                     traf_reprs.append(tr)
#             traf_reprs = torch.stack(traf_reprs, dim=0)  # B x d

#             # ---------------- compute losses ----------------
#             loss_contrast = contrast_traj_vs_traf(r1, traf_reprs, temperature=args.temperature) + contrast_traj_vs_traf(r2, traf_reprs, temperature=args.temperature)
#             loss_contrast = 0.5 * loss_contrast  # average both views

#             # optional MTP loss (trajectory segment prediction)
#             loss_mtp = torch.tensor(0.0, device=device)
#             if args.use_mtp:
#                 # mtp_logits: B x L x n_nodes
#                 # prepare targets: orig_arr padded, but we should set pad positions to -1 and ignore in CE
#                 targets = torch.LongTensor(orig_arr).to(device)  # B x L
#                 Bc, Lc, n_nodes = mtp_logits_orig.shape
#                 # flatten
#                 logits_flat = mtp_logits_orig.view(Bc*Lc, n_nodes)
#                 targets_flat = targets.view(Bc*Lc)
#                 # mask positions where orig_mask is False -> set target to -1 and ignore via criterion
#                 mask_flat = torch.from_numpy(orig_mask.reshape(-1)).to(device)
#                 targets_flat_masked = targets_flat.clone()
#                 targets_flat_masked[~mask_flat] = -1
#                 # use custom loss: compute CE only on valid positions
#                 valid_idx = mask_flat.nonzero(as_tuple=False).view(-1)
#                 if valid_idx.numel() > 0:
#                     logits_valid = logits_flat[valid_idx]
#                     targets_valid = targets_flat_masked[valid_idx]
#                     loss_mtp = ce_loss(logits_valid, targets_valid)
#                 else:
#                     loss_mtp = torch.tensor(0.0, device=device)

#             # combine losses
#             loss = args.alpha * loss_contrast + args.beta * loss_mtp

#             # backward
#             optim.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#             optim.step()

#             global_step += 1
#             pbar.set_postfix({'loss': float(loss.detach().cpu().item()),
#                                'contrast': float(loss_contrast.detach().cpu().item()),
#                                'mtp': float(loss_mtp.detach().cpu().item()) if isinstance(loss_mtp, torch.Tensor) else float(loss_mtp)})

#             # optional checkpointing
#             if global_step % args.save_every == 0:
#                 ckpt = {'model_state': model.state_dict(),
#                         'optim_state': optim.state_dict(),
#                         'step': global_step,
#                         'args': vars(args)}
#                 os.makedirs(args.ckpt_dir, exist_ok=True)
#                 torch.save(ckpt, os.path.join(args.ckpt_dir, f'pretrain_step_{global_step}.pt'))

#         # epoch end checkpoint
#         ckpt = {'model_state': model.state_dict(),
#                 'optim_state': optim.state_dict(),
#                 'step': global_step,
#                 'args': vars(args)}
#         os.makedirs(args.ckpt_dir, exist_ok=True)
#         torch.save(ckpt, os.path.join(args.ckpt_dir, f'pretrain_epoch_{epoch}.pt'))

#     print("Pretraining finished.")

# # ---------------- argument parsing ----------------
# def get_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--data_root', type=str, default='data', help='data root')
#     p.add_argument('--traj_file', type=str, default='data/trajectories.csv', help='trajectory csv')
#     p.add_argument('--edge_shp', type=str, default=None, help='edge shapefile (optional)')
#     p.add_argument('--static_file', type=str, default='data/static_nodes.npy', help='static features npy')
#     p.add_argument('--traffic_ts_file', type=str, default='data/traffic_timeseries.npy', help='traffic timeseries npy')
#     p.add_argument('--cache_dir', type=str, default='./cache', help='cache dir')
#     p.add_argument('--batch_size', type=int, default=8)
#     p.add_argument('--epochs', type=int, default=3)
#     p.add_argument('--lr', type=float, default=1e-4)
#     p.add_argument('--weight_decay', type=float, default=1e-5)
#     p.add_argument('--num_workers', type=int, default=4)
#     p.add_argument('--cpu', action='store_true')
#     # model params (must be consistent with your TrackMiniPaper args)
#     p.add_argument('--d_model', type=int, default=128)
#     p.add_argument('--n_heads', type=int, default=8)
#     p.add_argument('--traf_in_dim', type=int, default=2)
#     p.add_argument('--traffic_seq_len', type=int, default=12)
#     p.add_argument('--traj_enc_layers', type=int, default=2)
#     p.add_argument('--traf_enc_layers', type=int, default=2)
#     p.add_argument('--n_coatt_layers', type=int, default=1)
#     p.add_argument('--state_out_dim', type=int, default=2)
#     p.add_argument('--dropout', type=float, default=0.1)
#     # pretrain specifics
#     p.add_argument('--traj_max_len', type=int, default=32)
#     p.add_argument('--aug_dropout', type=float, default=0.2)
#     p.add_argument('--temperature', type=float, default=0.1)
#     p.add_argument('--alpha', type=float, default=1.0, help='weight for contrastive loss')
#     p.add_argument('--beta', type=float, default=0.1, help='weight for mtp loss')
#     p.add_argument('--use_mtp', action='store_true', help='enable MTP loss')
#     p.add_argument('--n_nodes', type=int, default=1000, help='vocab size (for mtp logits dim)')
#     p.add_argument('--traj_pool_sample', type=int, default=1, help='trajectories per sample to draw')
#     p.add_argument('--K_min', type=int, default=15)
#     p.add_argument('--num_time_bins', type=int, default=24)
#     p.add_argument('--khop_fallback', type=int, default=2)
#     p.add_argument('--grad_clip', type=float, default=5.0)
#     p.add_argument('--save_every', type=int, default=500)
#     p.add_argument('--ckpt_dir', type=str, default='./checkpoints')
#     p.add_argument('--force_recompute', action='store_true')
#     args = p.parse_args()
#     return args

# if __name__ == '__main__':
#     args = get_args()
#     # deterministic-ish
#     random.seed(42)
#     np.random.seed(42)
#     torch.manual_seed(42)
#     train(args)
