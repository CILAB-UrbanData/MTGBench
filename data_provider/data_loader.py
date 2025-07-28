import os
import numpy as np
import pandas as pd
import glob, random
import re
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt
from datetime import date, timedelta
from utils.tools import date_range
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
        taxi_seq = self.X_taxi[idx:idx+self.S]
        bike_seq = self.X_bike[idx:idx+self.S]
        A_taxi_seq = self.A_taxi[idx:idx+self.S]
        A_bike_seq = self.A_bike[idx:idx+self.S]
        label_taxi = self.Y_taxi[idx+self.S]
        label_bike = self.Y_bike[idx+self.S]
        return taxi_seq, bike_seq, A_taxi_seq, A_bike_seq, label_taxi, label_bike

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
        begin_time = pd.to_datetime('2008-05-31 03:50:00', format='%Y-%m-%d %H:%M:%S')
        end_time = pd.to_datetime('2008-06-10 02:10:00', format='%Y-%m-%d %H:%M:%S')
        self.flow = flow[(flow['index'] > begin_time) & (flow['index'] <= end_time)].set_index('index')

        self.on_epoch_start()  # 预生成样本

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
        T_start = random.randint(0, 599)  # 随机起始点
        for t in range(T_start, 1424, 600): # 每600分钟一个时间段,1424为总共时间段数
            aggregated_targets = []
            for seg in unique_segments:
                t_list = segment2targets[seg][0]
                aggregated_targets.append(t_list[t:t+6])
            targets.append(torch.stack(aggregated_targets, dim=0))
        targets_tensor = torch.stack(targets, dim=0)  

        inputs = {
            'segments': segments_tensor,
            'paths': padded,
            'forest': forest
        }
        inputs = forest
        
        return (inputs, T_start), targets_tensor

class SF20_forTrGNN_Dataset(Dataset):
    """
    SF20 for TrGNN Dataset
    """
    def __init__(self, args, flag, start_date, end_date, 
                 root_path, flow_path='sf_flow_100_trimmed.csv'):
        self.args = args
        self.root_path = root_path
        self.dates = date_range(start_date, end_date)
        self.flow_path = flow_path
        flow_df = pd.concat([pd.read_csv('fastdatasf/flow_%s_%s.csv'%(date, date), index_col=0) for date in self.dates])
        flow_df.columns = pd.Index(int(road_id) for road_id in flow_df.columns)
        self.start_date = dt.strptime(start_date, '%Y%m%d')
        self.end_date = dt.strptime(end_date, '%Y%m%d')

        date_list = [self.start_date + timedelta(days=i) for i in range((self.end_date - self.start_date).days + 1)]
        # 找出所有 weekday 的索引（周一到周五，weekday() 返回0~4）
        weekdays = np.array([i for i, d in enumerate(date_list) if d.weekday() < 5])

        assert flag in ['train', 'val', 'test']
        N_len = len(flow_df)
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
        flow_df = flow_df[idx_map[flag]]  # 按照flag划分数据集
        self.flow = flow_df.values

        