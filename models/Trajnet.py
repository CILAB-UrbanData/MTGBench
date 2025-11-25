import torch, os
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict
from layers.Spatial_Atten import SpatialAttention
from layers.Dilate_Causal_Con import DilatedCausalConv1d
import torch.nn.functional as F


class TemporalConvBranch(nn.Module):
    def __init__(self, in_channels, C, kernel_size=2, num_layers=3):
        """
        多层膨胀因果卷积：
          layer i 的 dilation = 2**i, i = 0...num_layers-1
        """
        super().__init__()
        layers = []
        for i in range(num_layers):
            d = 2 ** i
            ic = in_channels if i == 0 else C
            layers.append(
                DilatedCausalConv1d(ic, C, kernel_size=kernel_size, dilation=d)
            )
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, in_channels, T]
        return self.net(x)  # [N, C, T']

class SegmentEncoder(nn.Module):
    def __init__(self, 
                 in_channels=1,  # 单维度打点序列
                 C=16,
                 tau_P=6, tau_D=2, tau_W=2,
                 kernel_size=2, encoder_layers=3):
        super().__init__()
        self.branch = TemporalConvBranch(in_channels, C, kernel_size, encoder_layers)

    def forward(self, x_P, x_D, x_W):
        """
        batched 版本：
          x_P: [B, N, 1, tau_P]
          x_D: [B, N, 1, tau_D * tau_P]
          x_W: [B, N, 1, tau_W * tau_P]

        返回：
          h: [B, N, C, n_h]  （n_h = 3 条分支的时间长度总和）
        """
        B, N, _, TP = x_P.shape
        _, _, _, TD = x_D.shape
        _, _, _, TW = x_W.shape

        # 展平 batch 和节点维度，做一次 conv
        def encode_branch(x):
            # x: [B,N,1,T] -> [B*N,1,T]
            x = x.view(B * N, 1, x.size(-1))
            f = self.branch(x)              # [B*N, C, T_out]
            _, C, T_out = f.shape
            f = f.view(B, N, C, T_out)      # [B,N,C,T_out]
            return f, T_out

        f_P, Lp = encode_branch(x_P)
        f_D, Ld = encode_branch(x_D)
        f_W, Lw = encode_branch(x_W)

        # 沿时间维 concat -> [B,N,C,Lp+Ld+Lw] = [B,N,C,n_h]
        h = torch.cat([f_P, f_D, f_W], dim=-1)
        return h

class PropagatorWithForest(nn.Module):
    """
    利用前缀森林进行轨迹特征传播,末端同segment的表示取平均,输出包含segment id
    参考 Eq.(3) 扁平化版本
    """
    def __init__(self, n_h, n_channels):
        super().__init__()
        self.n_h = n_h
        self.C = n_channels
        self.D = n_h * n_channels
        self.proj_flat = nn.Linear(self.D, self.D, bias=False)

    def forward(self, segment_feats, forest_roots, attn_matrix):
        """
        Args:
          segment_feats: Tensor [NumofRoads, n_h, C]
          forest_roots:  List[TrieNode]
          attn_matrix:   Tensor [NumofRoads, NumofRoads]
        Returns:
          segments: Tensor [M], 对应每个输出表征的segment id
          z_end:     Tensor [M, n_h, C], M = 不同末端segment数量
        """
        NumofRoads, n_h, C = segment_feats.shape
        D = self.D
        feats_flat = segment_feats.reshape(NumofRoads, D)

        # 累积每个末端segment的z_flat列表
        z_acc = defaultdict(list)

        def dfs(node, path_ids, z_prev_flat=None):
            curr = path_ids[-1]
            h_flat = feats_flat[curr]
            if z_prev_flat is None:
                z_flat = h_flat
            else:
                Wz = self.proj_flat(z_prev_flat)
                prev = path_ids[-2] 
                alpha = attn_matrix[prev, curr]
                z_flat = alpha * Wz + h_flat
            # 如果到达子轨迹末端，累积
            if node.is_end:
                z_acc[curr].append(z_flat)
            # 继续遍历子分支
            for seg, child in node.children.items():
                dfs(child, path_ids + [seg], z_flat)

        # 遍历前缀森林
        for root in forest_roots:
            for seg, child in root.children.items():
                dfs(child, [seg], None)

        # 对每个末端segment求平均，并保留ID
        seg_ids, z_list = [], []
        for seg, zs in z_acc.items():
            stacked = torch.stack(zs, dim=0)          # [k, D]
            mean_flat = stacked.mean(dim=0)           # [D]
            z_list.append(mean_flat.view(n_h, C))
            seg_ids.append(seg)

        if z_list:
            segments = torch.tensor(seg_ids, dtype=torch.long, device=segment_feats.device)
            z_end = torch.stack(z_list, dim=0)       # [M, n_h, C]
        else:
            segments = torch.empty(0, dtype=torch.long, device=segment_feats.device)
            z_end = torch.empty(0, n_h, C, device=segment_feats.device)

        return segments, z_end

class Predictor(nn.Module):
    def __init__(self, pred_steps=1):
        super().__init__()
        self.fc = nn.LazyLinear(pred_steps)

    def forward(self, z_end):
        # features: [B, N, D]
        _, n_h, C = z_end.shape
        z_end = z_end.view(-1, n_h * C)  # 扁平化
        return self.fc(z_end)  # 输出 [B, pred_steps]

class Model(nn.Module):
    def __init__(self, args):
        #T1, T2, T3, NumofRoads, adj_mask, kernel_size=2, num_layers=3, outChannel_1=16
        super().__init__()
        self.encoder    = SegmentEncoder(1, args.outChannel_1, args.T1, args.T2, args.T3, args.kernel_size, args.encoder_layers)
        self.n_h = args.T1 * (1 + args.T2 + args.T3)  
        args.adj_mask = os.path.join(args.cache_dir, f'adjacency_{args.min_flow_count}_{args.data}.pkl')
        self.attention = SpatialAttention(args.NumofRoads, self.n_h, args.outChannel_1, args.adj_mask)  # [NumofRoads, NumofRoads]
        self.propagator = PropagatorWithForest(self.n_h, args.outChannel_1)
        self.predictor  = Predictor(pred_steps=args.T1)
        self.device = args.device
        self.__init_weights()  # 初始化权重

    def __init_weights(self):
        """
        初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, nn.LazyLinear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, inputs):
        forests = inputs['forest']                     # List[List[TrieNode]], len=B
        recents = inputs['recent'].to(self.device)     # [B,N,1,T1]
        dailys  = inputs['daily'].to(self.device)      # [B,N,1,T2*T1]
        weeklys = inputs['weekly'].to(self.device)     # [B,N,1,T3*T1]

        B, N, _, _ = recents.shape

        # 1) temporal encoder: 真正 batched
        features = self.encoder(recents, dailys, weeklys)  # [B,N,C,n_h]

        # 2) batched SpatialAttention: 得到 A_all: [B,N,N]
        A_all = self.attention(features)                  # 我们刚改好的 batched 版本

        preds_list = []
        segments_list = []

        # 3) 对每个样本独立用 prefix forest 做传播 & 预测
        for i in range(B):
            feat_i = features[i]       # [N,C,n_h]
            A_i    = A_all[i]          # [N,N]
            forest_i = forests[i]      # List[TrieNode]

            segs_i, z_end_i = self.propagator(feat_i, forest_i, A_i)  # segs_i: [M_i]
            preds_i = self.predictor(z_end_i)                         # [M_i,T1]

            preds_list.append(preds_i)
            segments_list.append(segs_i)
        
        

        return (preds_list , segments_list)
