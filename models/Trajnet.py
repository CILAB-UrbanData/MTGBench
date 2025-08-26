import torch, os
import torch.nn as nn
import pandas as pd
import numpy as np
from collections import defaultdict
from layers.Spatial_Atten import SpatialAttention
from layers.Dilate_Causal_Con import DilatedCausalConv1d
import torch.nn.functional as F
    #TODO 把时间上遍历所有flow数据的逻辑集成到model中


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
        # 三条时序分支
        self.branch = TemporalConvBranch(in_channels, C, kernel_size, encoder_layers)

    def forward(self, x_P, x_D, x_W):
        """
        输入：
          x_P: [n_s, 1, tau_P]  最近序列
          x_D: [n_s, 1, tau_D * tau_P]  日周期序列
          x_W: [n_s, 1, tau_W * tau_P]  周周期序列
        返回：
          enc: [n_s, C]
        """
        f_P = self.branch(x_P)   # -> [n_s, C, *]
        f_D = self.branch(x_D)   # -> [n_s, C, *]
        f_W = self.branch(x_W)   # -> [n_s, C, *]

        # 拼接并映射
        h = torch.cat([f_P, f_D, f_W], dim = -1)  # [n_s, C, n_h]

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
          segment_feats: Tensor [n_s, n_h, C]
          forest_roots:  List[TrieNode]
          attn_matrix:   Tensor [n_s, n_s]
        Returns:
          segments: Tensor [M], 对应每个输出表征的segment id
          z_end:     Tensor [M, n_h, C], M = 不同末端segment数量
        """
        n_s, n_h, C = segment_feats.shape
        D = self.D
        feats_flat = segment_feats.reshape(n_s, D)

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
        #T1, T2, T3, n_s, adj_mask, kernel_size=2, num_layers=3, outChannel_1=16
        super().__init__()
        self.encoder    = SegmentEncoder(1, args.outChannel_1, args.T1, args.T2, args.T3, args.kernel_size, args.encoder_layers)
        self.n_h = args.T1 * (1 + args.T2 + args.T3)  
        args.adj_mask = os.path.join(args.root_path, 'adjacency_trimmed.pkl')
        self.attention = SpatialAttention(args.n_s, self.n_h, args.outChannel_1, args.adj_mask)  # [n_s, n_s]
        self.propagator = PropagatorWithForest(self.n_h, args.outChannel_1)
        self.predictor  = Predictor(pred_steps=args.T1)
        self.device = args.device
        self.flow = os.path.join(args.root_path, 'sf_flow_100_trimmed.csv')
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

    def load_flow_data(self, T='2008-06-08 10:30:00', T1=6, T2=2, T3=2):
        """
        加载流量数据, 用于时间卷积
        """
        flow = pd.read_csv(self.flow, header=0)
        flow['index'] = pd.to_datetime(flow['index'])
        T = pd.to_datetime(T, format='%Y-%m-%d %H:%M:%S')
        
        recent = flow[
                    (flow['index'] >= T - pd.Timedelta(minutes=T1 * 10)) & 
                    (flow['index'] < T)]
        
        daily =  flow[
                    (flow['index'] >= T - pd.Timedelta(days=T2) - pd.Timedelta(minutes=T1 * 10) ) & 
                    (flow['index'] < T - pd.Timedelta(days=T2))]
        T2 -= 1    
        while T2 > 0:
            daily_part = flow[
                (flow['index'] >= T - pd.Timedelta(days=T2) - pd.Timedelta(minutes=T1 * 10)) &
                (flow['index'] < T - pd.Timedelta(days=T2))
            ]
            daily = pd.concat([daily, daily_part], ignore_index=True)
            T2 -= 1
        
        weekly =  flow[
                (flow['index'] >= T - pd.Timedelta(weeks=T3) - pd.Timedelta(minutes=T1 * 10) ) & 
                (flow['index'] < T - pd.Timedelta(weeks=T3))
                ]
        T3 -= 1    
        while T3 > 0:
            weekly_part =  flow[
                (flow['index'] >= T - pd.Timedelta(weeks=T3) - pd.Timedelta(minutes=T1 * 10) ) & 
                (flow['index'] < T - pd.Timedelta(weeks=T3))
            ]
            weekly = pd.concat([weekly, weekly_part], ignore_index=True)
            T3 -= 1

        # 只保留数值型列（去掉时间列）
        value_cols = [col for col in flow.columns if flow[col].dtype != 'datetime64[ns]']
        recent = recent[value_cols]
        daily = daily[value_cols]
        weekly = weekly[value_cols]

        return (
            torch.from_numpy(recent.values.astype(np.float32)).T,
            torch.from_numpy(daily.values.astype(np.float32)).T,
            torch.from_numpy(weekly.values.astype(np.float32)).T
        )   

    def forward(self, inputs):
        # recent/daily/weekly: [N, 1, T*], trajectories: [B, num_trajs, traj_len]
        forests = inputs['forest']
        recents = inputs['recent'].to(self.device)   # [time, N, 1, T*]
        dailys  = inputs['daily'].to(self.device)
        weeklys = inputs['weekly'].to(self.device)

        prediction = []
        for t in range(recents.shape[0]):  # 遍历 timerange
            recent  = recents[t]
            daily   = dailys[t]
            weekly  = weeklys[t]

            features = self.encoder(recent, daily, weekly)  # [N, C, L]
            A_matrix = self.attention(features)
            segments, z_end = self.propagator(features, forests, A_matrix)
            prediction.append(self.predictor(z_end))

        prediction = torch.stack(prediction, dim=0)
        return prediction