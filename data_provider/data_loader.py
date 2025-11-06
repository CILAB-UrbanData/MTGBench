import os
import numpy as np
import pandas as pd
import glob, random
import re
import torch
import pickle as pkl
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt
from datetime import date, timedelta
from utils.tools import date_range, to_sparse_tensor
import warnings

warnings.filterwarnings('ignore')
  
class MDTPRawloader(Dataset):
    def __init__(self, args, root_path, flag, normalization=True, S=24):
        super(MDTPRawloader, self).__init__()
        self.args = args
        self.root_path = root_path
        self.normalization = normalization
        self.S = S
        if self.normalization:
            self.data_path = "processed.npz"
        else:
            self.data_path = "processedwithoutnormalization.npz"
        path = os.path.join(self.root_path, self.data_path)
        data = np.load(path)
        #按照flag划分数据集
        assert flag in ['train', 'test', 'val']
        N_all = len(data['X_taxi'])
        train_n = int(N_all * 0.7)
        val_n   = int(N_all * 0.1)
        test_n  = N_all - train_n - val_n
        segments = [('train', train_n),
                ('val',   val_n),
                ('test',  test_n)]
        idx_map = {}
        start = 0
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        self.X_taxi, self.X_bike, self.A_taxi, self.A_bike, self.Y_taxi, self.Y_bike = (
            data['X_taxi'][idx_map[flag]], data['X_bike'][idx_map[flag]], data['A_taxi'][idx_map[flag]], data['A_bike'][idx_map[flag]], data['Y_taxi'][idx_map[flag]], data['Y_bike'][idx_map[flag]]
        )
    
    def __len__(self):
        return len(self.X_taxi) - self.S
    
    def __getitem__(self, idx):
        taxi_seq = torch.tensor(self.X_taxi[idx:idx+self.S], dtype=torch.float32)
        bike_seq = torch.tensor(self.X_bike[idx:idx+self.S], dtype=torch.float32)
        A_taxi_seq = torch.tensor(self.A_taxi[idx:idx+self.S], dtype=torch.float32)
        A_bike_seq = torch.tensor(self.A_bike[idx:idx+self.S], dtype=torch.float32)
        label_taxi = torch.tensor(self.Y_taxi[idx+self.S], dtype=torch.float32)
        label_bike = torch.tensor(self.Y_bike[idx+self.S], dtype=torch.float32)
        return taxi_seq, bike_seq, A_taxi_seq, A_bike_seq, label_taxi, label_bike
    
    def collate_fn(self, batch):
        taxi_seq, bike_seq, A_taxi, A_bike, y_taxi, y_bike = zip(*batch)
        taxi_seq = torch.stack(taxi_seq)
        bike_seq = torch.stack(bike_seq)
        A_taxi   = torch.stack(A_taxi)
        A_bike   = torch.stack(A_bike)
        y_taxi   = torch.stack(y_taxi)
        y_bike   = torch.stack(y_bike)

        # 这里自动拼接 target
        target = torch.cat([y_taxi, y_bike], dim=-1)

        return (taxi_seq, bike_seq, A_taxi, A_bike), target

class GaiyaForMDTP(Dataset):
    def __init__(self, args, root_path, flag, normalization=True, S=24):
        super(GaiyaForMDTP, self).__init__()
        self.args = args
        self.root_path = root_path
        self.normalization = normalization
        self.S = S
        if self.normalization:
            self.data_path = "processed.npz"
        else:
            self.data_path = "processedwithoutnormalization.npz"
        path = os.path.join(self.root_path, self.data_path)
        data = np.load(path)
        #按照flag划分数据集
        assert flag in ['train', 'test', 'val']
        N_all = len(data['X_taxi'])
        train_n = int(N_all * 0.7)
        val_n   = int(N_all * 0.1)
        test_n  = N_all - train_n - val_n
        segments = [('train', train_n),
                ('val',   val_n),
                ('test',  test_n)]
        idx_map = {}
        start = 0
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        self.X_taxi, self.A_taxi, self.Y_taxi = (
            data['X_taxi'][idx_map[flag]], data['A_taxi'][idx_map[flag]], data['Y_taxi'][idx_map[flag]]
        )
    
    def __len__(self):
        return len(self.X_taxi) - self.S
    
    def __getitem__(self, idx):
        taxi_seq = torch.tensor(self.X_taxi[idx:idx+self.S], dtype=torch.float32)
        A_taxi_seq = torch.tensor(self.A_taxi[idx:idx+self.S], dtype=torch.float32)
        label_taxi = torch.tensor(self.Y_taxi[idx+self.S], dtype=torch.float32)
        return taxi_seq, A_taxi_seq, label_taxi
    
    def collate_fn(self, batch):
        taxi_seq, A_taxi, y_taxi = zip(*batch)
        taxi_seq = torch.stack(taxi_seq)
        A_taxi   = torch.stack(A_taxi)
        target   = torch.stack(y_taxi)

        return (taxi_seq, A_taxi), target

# SF20_forTrajnet_Dataset
class TrieNode:
    """
    前缀树节点，用于批内前缀共享计算（可选）
    """
    def __init__(self):
        self.children = {}
        self.is_end = False

def build_prefix_forest(truncated_paths):
    """
    根据截断后子轨迹构建前缀森林
    返回根节点列表（仅在需要前缀优化时使用）
    """
    forest = {}
    for path in truncated_paths:
        root = forest.setdefault(path[0], TrieNode())
        node = root
        for seg in path:
            node = node.children.setdefault(seg, TrieNode())
        node.is_end = True
    return list(forest.values())

class SF20_forTrajnet_Dataset(Dataset):
    def __init__(self, args, flag, root_path,flow_path='sf_flow_100_trimmed.csv',
                 traj_path='raw_last_sf_100_win7_trimmed.csv',
                 trunc_length=7, samples_per_segment=5, batch_size=32):
        self.trunc_length = trunc_length
        self.samples_per_segment = samples_per_segment
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.samples_per_segment = samples_per_segment
        self.root_path = root_path

        # 构建 segment->轨迹映射
        self.seg2trajs = defaultdict(list)
        trajectories = pd.read_csv(os.path.join(self.root_path, traj_path), header=None).values.tolist()

        assert flag in ['train', 'val', 'test']
        N_len = len(trajectories)
        train_n = int(N_len * 0.7)
        val_n   = int(N_len * 0.1)
        test_n  = N_len - train_n - val_n
        segments = [('train', train_n),
                    ('val',   val_n),
                    ('test',  test_n)]
        idx_map = {}
        start = 0
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        trajectories = [trajectories[i] for i in idx_map[flag]]  # 按照flag划分数据集

        for traj in trajectories:
            self.seg2trajs[traj[-1]].append(traj)
        self.segments = [seg for seg, lst in self.seg2trajs.items() if lst]

        # 加载并过滤 flow 数据
        flow = pd.read_csv(os.path.join(self.root_path, flow_path), header=0)
        flow['index'] = pd.to_datetime(flow['index'])
        self.flow = flow

        self.on_epoch_start()  # 预生成样本

    def load_flow_data(self, T, T1=6, T2=2, T3=2):
        """
        和 Model 里的版本一致，只不过直接用 self.flow
        """
        flow = self.flow
        recent = flow[
            (flow['index'] >= T - pd.Timedelta(minutes=T1 * 10)) &
            (flow['index'] < T)
        ]
        
        daily = flow[
            (flow['index'] >= T - pd.Timedelta(days=T2) - pd.Timedelta(minutes=T1 * 10)) &
            (flow['index'] < T - pd.Timedelta(days=T2))
        ]
        T2 -= 1
        while T2 > 0:
            daily_part = flow[
                (flow['index'] >= T - pd.Timedelta(days=T2) - pd.Timedelta(minutes=T1 * 10)) &
                (flow['index'] < T - pd.Timedelta(days=T2))
            ]
            daily = pd.concat([daily, daily_part], ignore_index=True)
            T2 -= 1

        weekly = flow[
            (flow['index'] >= T - pd.Timedelta(weeks=T3) - pd.Timedelta(minutes=T1 * 10)) &
            (flow['index'] < T - pd.Timedelta(weeks=T3))
        ]
        T3 -= 1
        while T3 > 0:
            weekly_part = flow[
                (flow['index'] >= T - pd.Timedelta(weeks=T3) - pd.Timedelta(minutes=T1 * 10)) &
                (flow['index'] < T - pd.Timedelta(weeks=T3))
            ]
            weekly = pd.concat([weekly, weekly_part], ignore_index=True)
            T3 -= 1

        # 去掉 datetime 列
        value_cols = [col for col in flow.columns if flow[col].dtype != 'datetime64[ns]']
        recent = torch.from_numpy(recent[value_cols].values.astype(np.float32)).T
        daily  = torch.from_numpy(daily[value_cols].values.astype(np.float32)).T
        weekly = torch.from_numpy(weekly[value_cols].values.astype(np.float32)).T

        return recent, daily, weekly

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def on_epoch_start(self):
        # 预生成样本
        self.samples = []
        for seg in self.segments:
            candidates = self.seg2trajs[seg]
            # 采样或重复采样
            if len(candidates) >= self.samples_per_segment:
                chosen = random.sample(candidates, self.samples_per_segment)
            else:
                chosen = candidates

            flow_values = torch.tensor(self.flow[str(seg)].values, dtype=torch.float)
            for traj in chosen:
                raw_paths = traj
                tensor_paths = torch.tensor(traj, dtype=torch.long)
                self.samples.append((seg, raw_paths, tensor_paths, flow_values))
   
    def collate_fn(self, batch):
        """
        将多组(sample)合并成一个mini-batch：
        - 构建前缀森林
        - paths pad到同一长度
        - 按 segment 聚合目标 flow 向量
        """
        segments, raw_paths, tensor_paths, targets = zip(*batch)

        # 构建前缀森林
        forest = build_prefix_forest(raw_paths)

        # pad tensor_paths
        padded = pad_sequence(list(tensor_paths), batch_first=True, padding_value=0)

        segments_tensor = torch.tensor(segments, dtype=torch.long)

        # 聚合 targets: 按 segment 聚合（平均）
        segment2targets = defaultdict(list)
        for seg, target in zip(segments, targets):
            segment2targets[seg].append(target)

        # 最终的 targets 要按照 forward() 输出顺序一致：去重后保序
        unique_segments = list(dict.fromkeys(segments))  # 去重且保持顺序
        targets = []
        T_start = random.randint(0, 59)  # 随机起始点
        for t in range(T_start, 1424, 60): # 每60分钟一个时间段,1424为总共时间段数
            aggregated_targets = []
            for seg in unique_segments:
                t_list = segment2targets[seg][0]
                aggregated_targets.append(t_list[t:t+6])
            targets.append(torch.stack(aggregated_targets, dim=0))
        targets_tensor = torch.stack(targets, dim=0)  

        T_start_time = str(pd.to_datetime('2008-05-31 04:00', format='%Y-%m-%d %H:%M') + pd.Timedelta(minutes=T_start * 10))
        timerange = pd.date_range(T_start_time, '2008-06-10 01:10', freq='600min')

        recents, dailys, weeklys = [], [], []
        for t in timerange:
            recent, daily, weekly = self.load_flow_data(T=t)
            recents.append(recent.unsqueeze(1))
            dailys.append(daily.unsqueeze(1))
            weeklys.append(weekly.unsqueeze(1))

        recents  = torch.stack(recents, dim=0)   # [time, N, 1, T*]
        dailys   = torch.stack(dailys, dim=0)
        weeklys  = torch.stack(weeklys, dim=0)

        inputs = {
            'segments': segments_tensor,
            'paths': padded,
            'forest': forest,
            'recent': recents,
            'daily': dailys,
            'weekly': weeklys
        }
        return inputs, targets_tensor

# SF20_forTrGNN_Dataset
class SF20_forTrGNN_Dataset(Dataset):
    """
    SF20 for TrGNN Dataset
    """
    def __init__(self, args, flag, start_date, end_date, root_path):
        self.args = args
        self.root_path = root_path
        self.dates = date_range(start_date, end_date)
        preprocess_path = os.path.join(self.root_path, 'cache/preprocess_TrGNNsf_20.pkl')

        # weekdays scaler都要有 
        flow_df = pd.concat([pd.read_csv(os.path.join(root_path, 'flow_%s_%s.csv'%(date, date)), index_col=0) for date in self.dates])
        flow_df.columns = pd.Index(int(road_id) for road_id in flow_df.columns)
        self.start_date = dt.strptime(start_date, '%Y%m%d')
        self.end_date = dt.strptime(end_date, '%Y%m%d')

        date_list = [self.start_date + timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]
        # 找出所有 weekday 的索引（周一到周五，weekday() 返回0~4）
        self.weekdays = np.array([i for i, d in enumerate(date_list) if d.weekday() < 5])

        assert flag in ['train', 'val', 'test']
        N_len = int(len(flow_df) * 23 / 24)  # 只保留23小时的数据
        train_n = int(N_len * 0.7)
        val_n   = int(N_len * 0.1)
        test_n  = N_len - train_n - val_n
        segments = [('train', train_n),
                    ('val',   val_n),
                    ('test',  test_n)]
        idx_map = {}
        start = 0
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        scaler = StandardScaler().fit(
            flow_df.iloc[idx_map['train'] + idx_map['val']].values
            ) # normalize flow
        self.scaler = scaler
        
        try:
            print('Loading preprocessed data...')
            with open(preprocess_path, 'rb') as f:
                normalized_flows, transitions_ToD, W, W_norm = pkl.load(f)
        except FileNotFoundError:
            print("文件名有错或者还未进行预处理！")      
            
        self.normalized_flows = normalized_flows
        self.transitions_ToD = transitions_ToD
        self.idx_map = idx_map[flag]
        self.flow = flow_df
    
    def __len__(self):
        return len(self.idx_map)
    
    def __getitem__(self, idx):
        i = self.idx_map[idx]
        d = i // 92
        t = i % 92       

        X = self.normalized_flows[d*96+t: d*96+t+self.args.seq_len]
        T = tuple(self.transitions_ToD[t: t+self.args.seq_len])
        y_true = self.normalized_flows[d*96+t+self.args.seq_len]

        ToD = torch.from_numpy(np.eye(24)[np.full((self.flow.shape[1]), ((t+4) * 15 // 60) % 24)]).float() # one-hot encoding: hour of day. (n_road, 24)
        DoW = torch.from_numpy(np.full((self.flow.shape[1], 1), int(d in self.weekdays))).float() # indicator: 1 for weekdays, 0 for weekends/PHs. (n_road, 1)

        return X, T, ToD, DoW, y_true
    
    def collate_fn(self, batch):
        """
        batch: list of samples, 每个 sample=(X, T, ToD, DoW, y_true)
        - X:      Tensor (seq_len, n_road)
        - T:      tuple of length H of sparse (n_road,n_road)
        - ToD:    Tensor (n_road, 24)
        - DoW:    Tensor (n_road, 1)
        - y_true: Tensor (n_road,)
        """
        # unzip
        Xs, Ts, ToDs, DoWs, ys = zip(*batch)
        
        # 1) stack X, ToD, DoW, y_true
        X_batch   = torch.stack(Xs,   dim=0)  # (B, H, n_road)
        ToD_batch = torch.stack(ToDs, dim=0)  # (B, n_road, 24)
        DoW_batch = torch.stack(DoWs, dim=0)  # (B, n_road, 1)
        y_batch   = torch.stack(ys,   dim=0)  # (B, n_road)
        
        # 2) 处理 T: 直接转换为 dense tensor
        H = len(Ts[0])
        B = len(Ts)
        
        # 假设所有稀疏矩阵都有相同的形状 (n_road, n_road)
        n_road = Ts[0][0].shape[0]
        
        T_batch = torch.zeros(B, H, n_road, n_road)
        for b in range(B):
            for t in range(H):
                T_batch[b, t] = Ts[b][t].to_dense()
        
        # 3) 打包 input
        inp = {
            'X':   X_batch,
            'T':   T_batch, #(H, B, n_road, n_road)
            'ToD': ToD_batch,
            'DoW': DoW_batch,
        }
        return inp, y_batch

class DiDi_forTrGNN_Dataset(Dataset):

    def __init__(self, args, flag, start_date, end_date, root_path):
        self.args = args
        self.root_path = root_path
        self.dates = date_range(start_date, end_date)
        preprocess_path = os.path.join(self.root_path, 'cache/preprocess_DiDiTrGNN.pkl')

        # weekdays scaler都要有 
        flow_df = pd.concat([pd.read_csv(os.path.join(root_path, 'tmp/flow_matched_%s_dedup_with_dwell.csv'%(date)), index_col=0) for date in self.dates])
        flow_df.columns = pd.Index(int(road_id) for road_id in flow_df.columns)
        self.start_date = dt.strptime(start_date, '%Y%m%d')
        self.end_date = dt.strptime(end_date, '%Y%m%d')

        date_list = [self.start_date + timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]
        # 找出所有 weekday 的索引（周一到周五，weekday() 返回0~4）
        self.weekdays = np.array([i for i, d in enumerate(date_list) if d.weekday() < 5])

        assert flag in ['train', 'val', 'test']
        N_len = int(len(flow_df) * 23 / 24)  # 只保留23小时的数据
        train_n = int(N_len * 0.7)
        val_n   = int(N_len * 0.1)
        test_n  = N_len - train_n - val_n
        segments = [('train', train_n),
                    ('val',   val_n),
                    ('test',  test_n)]
        idx_map = {}
        start = 0
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        scaler = StandardScaler().fit(
            flow_df.iloc[idx_map['train'] + idx_map['val']].values
            ) # normalize flow
        self.scaler = scaler
        
        try:
            print('Loading preprocessed data...')
            with open(preprocess_path, 'rb') as f:
                normalized_flows, transitions_ToD, W, W_norm = pkl.load(f)
        except FileNotFoundError:
            print("文件名有错或者还未进行预处理！")      
            
        self.normalized_flows = normalized_flows
        self.transitions_ToD = transitions_ToD
        self.idx_map = idx_map[flag]
        self.flow = flow_df
    
    def __len__(self):
        return len(self.idx_map)
    
    def __getitem__(self, idx):
        i = self.idx_map[idx]
        d = i // 92
        t = i % 92       

        X = self.normalized_flows[d*96+t: d*96+t+self.args.seq_len]
        T = tuple(self.transitions_ToD[t: t+self.args.seq_len])
        y_true = self.normalized_flows[d*96+t+self.args.seq_len]

        ToD = torch.from_numpy(np.eye(24)[np.full((self.flow.shape[1]), ((t+4) * 15 // 60) % 24)]).float() # one-hot encoding: hour of day. (n_road, 24)
        DoW = torch.from_numpy(np.full((self.flow.shape[1], 1), int(d in self.weekdays))).float() # indicator: 1 for weekdays, 0 for weekends/PHs. (n_road, 1)

        return X, T, ToD, DoW, y_true
    
    def collate_fn(self, batch):
        """
        batch: list of samples, 每个 sample=(X, T, ToD, DoW, y_true)
        - X:      Tensor (seq_len, n_road)
        - T:      tuple of length H of sparse (n_road,n_road)
        - ToD:    Tensor (n_road, 24)
        - DoW:    Tensor (n_road, 1)
        - y_true: Tensor (n_road,)
        """
        # unzip
        Xs, Ts, ToDs, DoWs, ys = zip(*batch)
        
        # 1) stack X, ToD, DoW, y_true
        X_batch   = torch.stack(Xs,   dim=0)  # (B, H, n_road)
        ToD_batch = torch.stack(ToDs, dim=0)  # (B, n_road, 24)
        DoW_batch = torch.stack(DoWs, dim=0)  # (B, n_road, 1)
        y_batch   = torch.stack(ys,   dim=0)  # (B, n_road)
        
        # 2) 处理 T: 直接转换为 dense tensor
        H = len(Ts[0])
        B = len(Ts)
        
        # 假设所有稀疏矩阵都有相同的形状 (n_road, n_road)
        n_road = Ts[0][0].shape[0]
        
        T_batch = torch.zeros(B, H, n_road, n_road)
        for b in range(B):
            for t in range(H):
                T_batch[b, t] = Ts[b][t].to_dense()
        
        # 3) 打包 input
        inp = {
            'X':   X_batch,
            'T':   T_batch, #(H, B, n_road, n_road)
            'ToD': ToD_batch,
            'DoW': DoW_batch,
        }
        return inp, y_batch

class DiDi_forTrajnet_Dataset(Dataset):
    def __init__(self, args, flag, root_path,flow_path='merged.csv',
                 traj_path='final_traj_dataset.csv',
                 trunc_length=7, samples_per_segment=5, batch_size=32):
        self.trunc_length = trunc_length
        self.samples_per_segment = samples_per_segment
        self.batch_size = batch_size
        self.trunc_length = trunc_length
        self.samples_per_segment = samples_per_segment
        self.root_path = root_path
        self.args = args

        # 构建 segment->轨迹映射
        self.seg2trajs = defaultdict(list)
        trajectories = pd.read_csv(os.path.join(self.root_path, traj_path), header=None,  quotechar='"', skipinitialspace=True).values.tolist()

        assert flag in ['train', 'val', 'test']
        N_len = len(trajectories)
        train_n = int(N_len * 0.7)
        val_n   = int(N_len * 0.1)
        test_n  = N_len - train_n - val_n
        segments = [('train', train_n),
                    ('val',   val_n),
                    ('test',  test_n)]
        idx_map = {}
        start = 0
        for name, length in segments:
            idx_map[name] = list(range(start, start + length))
            start += length
        trajectories = [trajectories[i] for i in idx_map[flag]]  # 按照flag划分数据集

        for traj in trajectories:
            # Fix: Handle different trajectory formats
            if isinstance(traj, list):
                if len(traj) > 0:
                    segment_id = traj[-1]  # Last element is segment ID
                    self.seg2trajs[segment_id].append(traj)
            else:
                print(f"Warning: Unexpected trajectory format: {type(traj)}")

        self.segments = [seg for seg, lst in self.seg2trajs.items() if lst]

        # 加载并过滤 flow 数据
        flow = pd.read_csv(os.path.join(self.root_path, flow_path), header=0)
        flow['time_bin'] = pd.to_datetime(flow['time_bin'])
        self.flow = flow

        self.on_epoch_start()  # 预生成样本

    def load_flow_data(self, T, T1=6, T2=2, T3=2):
        """
        和 Model 里的版本一致，只不过直接用 self.flow
        """
        flow = self.flow
        recent = flow[
            (flow['time_bin'] >= T - pd.Timedelta(minutes=T1 * 10)) &
            (flow['time_bin'] < T)
        ]
        
        daily = flow[
            (flow['time_bin'] >= T - pd.Timedelta(days=T2) - pd.Timedelta(minutes=T1 * 10)) &
            (flow['time_bin'] < T - pd.Timedelta(days=T2))
        ]
        T2 -= 1
        while T2 > 0:
            daily_part = flow[
                (flow['time_bin'] >= T - pd.Timedelta(days=T2) - pd.Timedelta(minutes=T1 * 10)) &
                (flow['time_bin'] < T - pd.Timedelta(days=T2))
            ]
            daily = pd.concat([daily, daily_part], ignore_index=True)
            T2 -= 1

        weekly = flow[
            (flow['time_bin'] >= T - pd.Timedelta(weeks=T3) - pd.Timedelta(minutes=T1 * 10)) &
            (flow['time_bin'] < T - pd.Timedelta(weeks=T3))
        ]
        T3 -= 1
        while T3 > 0:
            weekly_part = flow[
                (flow['time_bin'] >= T - pd.Timedelta(weeks=T3) - pd.Timedelta(minutes=T1 * 10)) &
                (flow['time_bin'] < T - pd.Timedelta(weeks=T3))
            ]
            weekly = pd.concat([weekly, weekly_part], ignore_index=True)
            T3 -= 1

        # 去掉 datetime 列
        value_cols = [col for col in flow.columns if flow[col].dtype != 'datetime64[ns]']
        recent = torch.from_numpy(recent[value_cols].values.astype(np.float32)).T
        daily  = torch.from_numpy(daily[value_cols].values.astype(np.float32)).T
        weekly = torch.from_numpy(weekly[value_cols].values.astype(np.float32)).T

        return recent, daily, weekly

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def on_epoch_start(self):
        # 预生成样本
        self.samples = []
        for seg in self.segments:
            candidates = self.seg2trajs[seg]
            # 采样或重复采样
            if len(candidates) >= self.samples_per_segment:
                chosen = random.sample(candidates, self.samples_per_segment)
            else:
                chosen = candidates

            # 修复：确保flow数据访问正确，处理segment ID中的特殊字符
            try:
                # 清理segment ID，去除可能的引号和空白字符
                clean_seg = str(seg).strip().strip('"').strip("'")
                
                if clean_seg in self.flow.columns:
                    flow_values = torch.tensor(self.flow[clean_seg].values, dtype=torch.float)
                elif str(seg) in self.flow.columns:
                    flow_values = torch.tensor(self.flow[str(seg)].values, dtype=torch.float)
                else:
                    # 如果segment列不存在，尝试查找相似的列名
                    available_cols = [col for col in self.flow.columns if str(seg) in str(col) or clean_seg in str(col)]
                    if available_cols:
                        print(f"使用相似的列: {available_cols[0]} 替代 {seg}")
                        flow_values = torch.tensor(self.flow[available_cols[0]].values, dtype=torch.float)
                    else:
                        print(f"警告：在flow数据中未找到segment {seg} (清理后: {clean_seg})")
                        print(f"可用的列: {list(self.flow.columns)[:10]}...")  # 显示前10个列名
                        flow_values = torch.zeros(len(self.flow), dtype=torch.float)
            except Exception as e:
                print(f"为segment {seg}加载flow数据时出错: {e}")
                flow_values = torch.zeros(100, dtype=torch.float)  # 虚拟值

            for traj in chosen:
                raw_paths = traj
                # 修复：确保轨迹数据在张量转换前格式正确
                try:
                    # 确保traj是整数列表
                    if isinstance(traj, list):
                        # 将任何字符串元素转换为整数
                        clean_traj = []
                        for item in traj:
                            if isinstance(item, str):
                                item = item.strip().strip('"').strip("'")
                                try:
                                    clean_traj.append(int(item))
                                except ValueError:
                                    print(f"警告：无法将 '{item}' 转换为整数")
                                    continue
                            else:
                                clean_traj.append(int(item))
                        tensor_paths = torch.tensor(clean_traj, dtype=torch.long)
                    else:
                        print(f"警告：意外的轨迹类型: {type(traj)}")
                        continue
                        
                except Exception as e:
                    print(f"将轨迹转换为张量时出错: {e}")
                    print(f"轨迹数据: {traj}")
                    continue
                self.samples.append((seg, raw_paths, tensor_paths, flow_values))
   
    def collate_fn(self, batch):
        """
        将多组(sample)合并成一个mini-batch：
        - 构建前缀森林
        - paths pad到同一长度
        - 按 segment 聚合目标 flow 向量
        """
        segments, raw_paths, tensor_paths, targets = zip(*batch)

        # 构建前缀森林
        forest = build_prefix_forest(raw_paths)

        # pad tensor_paths
        padded = pad_sequence(list(tensor_paths), batch_first=True, padding_value=0)

        segments_tensor = torch.tensor(segments, dtype=torch.long)

        # 聚合 targets: 按 segment 聚合（平均）
        segment2targets = defaultdict(list)
        for seg, target in zip(segments, targets):
            segment2targets[seg].append(target)

        # 最终的 targets 要按照 forward() 输出顺序一致：去重后保序
        unique_segments = list(dict.fromkeys(segments))  # 去重且保持顺序
        targets = []
        T_start = random.randint(0, 49)  # 随机起始点
        for t in range(T_start, 709, 50): # 每10分钟一个时间段,709为总共时间段数
            aggregated_targets = []
            for seg in unique_segments:
                t_list = segment2targets[seg][0]
                aggregated_targets.append(t_list[t:t+6])
            targets.append(torch.stack(aggregated_targets, dim=0))
        targets_tensor = torch.stack(targets, dim=0)  

        T_start_time = str(pd.to_datetime('2016-11-03 01:00', format='%Y-%m-%d %H:%M') + pd.Timedelta(minutes=T_start * 10))
        timerange = pd.date_range(T_start_time, '2016-11-07 23:00', freq='500min')

        recents, dailys, weeklys = [], [], []
        for t in timerange:
            recent, daily, weekly = self.load_flow_data(T=t)
            recents.append(recent.unsqueeze(1))
            dailys.append(daily.unsqueeze(1))
            weeklys.append(weekly.unsqueeze(1))

        recents  = torch.stack(recents, dim=0)   # [time, N, 1, T*]
        dailys   = torch.stack(dailys, dim=0)
        weeklys  = torch.stack(weeklys, dim=0)

        inputs = {
            'segments': segments_tensor,
            'paths': padded,
            'forest': forest,
            'recent': recents,
            'daily': dailys,
            'weekly': weeklys
        }
        return inputs, targets_tensor