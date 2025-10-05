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
    loss_history = []

    for epoch in range(args.pretrain_epochs):
        running = 0.0
        pbar = tqdm(dataloader, desc=f"Pretrain {epoch+1}/{args.pretrain_epochs}")
        for batch_list in pbar:
            # dataset returns list-of-items from collate_fn; we use the first item for synthetic demo
            item = batch_list[0]
            # required fields from dataset
            edge_index = item['edge_index'].to(device)   # 2 x E (LongTensor)
            P_edge = item['P_edge'].to(device)           # E (FloatTensor)
            static = item['static'].to(device)           # N x C
            S_hist = item['S_hist'].to(device)           # T x N x C_state
            weekly_idx = item.get('weekly_idx', None)
            daily_idx = item.get('daily_idx', None)
            if weekly_idx is not None:
                weekly_idx = weekly_idx.to(device)
            if daily_idx is not None:
                daily_idx = daily_idx.to(device)
            traj_pool = item['trajs']  # list of (nodes, times)

            # ========== 1) compute separate view representations ==========
            # H_traj: strict static-only per paper eq(1)
            H_traj = model.compute_H_traj(static, edge_index)  # N x d
            # H_traf: static + traffic history
            H_traf = model.compute_H_traf(static, S_hist, edge_index, P_edge=P_edge, weekly_idx=weekly_idx, daily_idx=daily_idx)

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
            H_masked = model.compute_H_traf(static, S_hist_masked, edge_index, P_edge=P_edge, weekly_idx=weekly_idx, daily_idx=daily_idx)
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
