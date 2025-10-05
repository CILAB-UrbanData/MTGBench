# tasks/finetune.py
"""
Finetune 脚本（已替换） — 关键点：
- 若想让下游也受 co-att 教导，将 co-att 放在 compute_H_traj & compute_H_traf 之后再预测
- 若下游仅依赖 traffic view，也可以跳过 co-att（本脚本默认执行 co-att）
"""

import torch
import numpy as np
import os
from tqdm import tqdm
from utils.metrics import next_state_loss

# 与 pretrain 中相同的反向 P_edge 构造函数（可放到 utils 里）
def build_reverse_P(edge_index, P_edge, default_val=-10.0):
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    E = src.shape[0]
    pair_to_val = {}
    for i in range(E):
        pair_to_val[(int(src[i]), int(dst[i]))] = float(P_edge[i].cpu().item()) if P_edge is not None else float(default_val)
    rev_vals = []
    for i in range(E):
        rev_pair = (int(dst[i]), int(src[i]))
        if rev_pair in pair_to_val:
            rev_vals.append(pair_to_val[rev_pair])
        else:
            rev_vals.append(float(default_val))
    return torch.tensor(rev_vals, dtype=P_edge.dtype, device=P_edge.device)

def run_finetune(model, dataloader, optimizer, args, load_pretrained=True):
    device = args.device
    if load_pretrained and os.path.exists('checkpoints/pretrained.pth'):
        model.load_state_dict(torch.load('checkpoints/pretrained.pth', map_location=device))
        print("Loaded pretrained.pth")

    model.train()
    loss_list = []
    for epoch in range(args.finetune_epochs):
        running = 0.0
        pbar = tqdm(dataloader, desc=f"Finetune {epoch+1}/{args.finetune_epochs}")
        for batch_list in pbar:
            item = batch_list[0]
            edge_index = item['edge_index'].to(device)
            P_edge = item['P_edge'].to(device)
            static = item['static'].to(device)
            S_hist = item['S_hist'].to(device)
            weekly_idx = item.get('weekly_idx', None)
            daily_idx = item.get('daily_idx', None)
            if weekly_idx is not None:
                weekly_idx = weekly_idx.to(device)
            if daily_idx is not None:
                daily_idx = daily_idx.to(device)

            # compute strict H_traj and H_traf
            H_traj = model.compute_H_traj(static, edge_index)
            H_traf = model.compute_H_traf(static, S_hist, edge_index, P_edge=P_edge, weekly_idx=weekly_idx, daily_idx=daily_idx)

            # prepare co-att edges & reverse P mapping
            edge_index_traf2traj = edge_index
            edge_index_traj2traf = torch.stack([edge_index[1], edge_index[0]], dim=0).to(edge_index.device)
            P_edge_traf2traj = P_edge
            P_edge_traj2traf = build_reverse_P(edge_index, P_edge)

            # co-att (we call it here so H_traf used for downstream has absorbed traj view)
            H_traj, H_traf = model.co_attention_exchange(H_traj, H_traf,
                                                         edge_index_traf2traj=edge_index_traf2traj,
                                                         edge_index_traj2traf=edge_index_traj2traf,
                                                         P_edge_traf2traj=P_edge_traf2traj,
                                                         P_edge_traj2traf=P_edge_traj2traf)

            # downstream prediction from H_traf (co-attended)
            pred_next = model.predict_next_state(H_traf)
            true_next = S_hist[-1].to(device)
            loss = next_state_loss(pred_next, true_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        epoch_loss = running / max(1, len(dataloader))
        loss_list.append(epoch_loss)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/finetuned.pth')
    np.save('checkpoints/loss_finetune.npy', np.array(loss_list))
    print("Finetune done: checkpoints/finetuned.pth, loss_finetune.npy saved")
