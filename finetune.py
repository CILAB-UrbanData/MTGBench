# tasks/finetune.py
"""
Finetune 脚本（已替换） — 关键点：
- 若想让下游也受 co-att 教导，将 co-att 放在 compute_H_traj & compute_H_traf 之后再预测
- 若下游仅依赖 traffic view，也可以跳过 co-att（本脚本默认执行 co-att）
"""

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from utils.metrics import next_state_loss
from data_provider.track_data import SegmentRoadDataset, batch_collate_fn
from models.TRACK import Model
import argparse

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
    model.to(device)
    model.train()
    loss_list = []
    for epoch in range(args.finetune_epochs):
        running = 0.0
        pbar = tqdm(dataloader, desc=f"Finetune {epoch+1}/{args.finetune_epochs}")
        for batch in pbar:
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

            # downstream prediction from H_traf (co-attended)
            pred_next = model.predict_next_state(H_traf)
            true_next = S_hist[:,-1].to(device)
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

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--static_feat_dim', type=int, default=3)
    argparser.add_argument('--d_model', type=int, default=32, help='Dimension of the model')
    argparser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    argparser.add_argument('--traffic_seq_len', type=int, default=7)
    argparser.add_argument('--n_nodes', type=int, default=26659)
    argparser.add_argument('--finetune_epochs', type=int, default=50, help='Number of finetuning epochs')
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
    dataset = SegmentRoadDataset("data/sf_data/raw")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=batch_collate_fn, drop_last=True)
    N = int(dataset.static.shape[0])  
    has_unk = (len(getattr(dataset, "idx2seg", [])) > 0 and dataset.idx2seg[-1] == "UNK")
    N_graph = N - 1 if has_unk else N   
    args.n_nodes = N    
    print(f"[pretrain] N_total={N}, N_graph={N_graph}, has_unk={has_unk}")

    model = Model(args)  # Initialize your model with appropriate args
    optimizer = Adam(model.parameters(), lr=0.001)

    run_finetune(model, dataloader, optimizer, args, load_pretrained=True)
