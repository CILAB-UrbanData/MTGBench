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