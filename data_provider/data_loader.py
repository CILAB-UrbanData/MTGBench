import os
import numpy as np
import pandas as pd
import glob, random, pickle
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
from scipy.interpolate import interp1d
from tqdm import tqdm
from geographiclib.geodesic import Geodesic
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



class DiffTrajProcess:
    def __init__(self, config: dict):
        required_keys = ["traj_length", "grid_size", "input_csv", "output_dir", "min_points"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置缺少必要键：{key}")
        
        self.config = config
        self.output_dir = config["output_dir"]
        self.df = None
        self.grouped_orders = None
        self.geo_bounds = None
        self.traj_array = None
        self.head_array = None
        self.traj_norm_params = None
        self.head_scaler = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"输出目录已创建：{os.path.abspath(self.output_dir)}")
        print(f"目标配置：轨迹长度={config['traj_length']}，网格尺寸={config['grid_size']}x{config['grid_size']}")


    @staticmethod
    def haversine_distance(lng1: float, lat1: float, lng2: float, lat2: float) -> float:
        try:
            return Geodesic.WGS84.Inverse(lat1, lng1, lat2, lng2)['s12']
        except Exception:
            R = 6371000
            phi1, phi2 = np.radians(lat1), np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lng2 - lng1)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
            return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))


    @staticmethod
    def timestamp_to_depature_index(dt) -> int:
        total_minutes = dt.hour * 60 + dt.minute
        return total_minutes // 5


    def read_and_clean_data(self) -> None:   
        read_csv_args = {
            "names": ['司机ID', '订单ID', 'GPS时间', '经度', '纬度'],
            "dtype": {
                '司机ID': str,
                '订单ID': str,
                '经度': str,
                '纬度': str,
                'GPS时间': str
            },
            "chunksize": 100000,
            "encoding": "utf-8"
        }
        if pd.__version__ >= "1.3.0":
            read_csv_args["on_bad_lines"] = "skip"
        else:
            read_csv_args["error_bad_lines"] = False

        chunks = []
        for chunk in tqdm(
            pd.read_csv(self.config["input_csv"], **read_csv_args),
            desc="分块读取CSV"
        ):
            chunks.append(chunk)
        
        self.df = pd.concat(chunks, ignore_index=True)
        raw_count = len(self.df)
        print(f"原始数据总行数：{raw_count:,}")

        self.df['GPS时间'] = pd.to_datetime(self.df['GPS时间'], errors='coerce')
        self.df['经度'] = pd.to_numeric(self.df['经度'], errors='coerce')
        self.df['纬度'] = pd.to_numeric(self.df['纬度'], errors='coerce')
        self.df = self.df.dropna(subset=['订单ID', 'GPS时间', '经度', '纬度'])
        self.df = self.df[(self.df['经度'] > 0) & (self.df['纬度'] > 0)]
        
        clean_count = len(self.df)
        print(f"清洗后保留行数：{clean_count:,}（过滤率：{1 - clean_count/raw_count:.2%}）")


    def group_and_filter_orders(self) -> None:
        print("\n" + "="*50)
        print("="*50)
        
        grouped = self.df.groupby('订单ID')
        print(f"原始订单总数：{len(grouped)}")

        min_points = self.config["min_points"]
        valid_orders = [oid for oid, group in grouped if len(group) >= min_points]
        self.df = self.df[self.df['订单ID'].isin(valid_orders)]
        self.grouped_orders = self.df.groupby('订单ID')
        
        valid_count = len(self.grouped_orders)        
        print(f"有效订单数：{valid_count}（过滤短轨迹：{len(grouped) - valid_count}个）")


    def calculate_geo_range(self) -> None:
        all_lng = []
        all_lat = []
        for _, group in self.grouped_orders:
            all_lng.extend(group['经度'].values)
            all_lat.extend(group['纬度'].values)
        all_lng = np.array(all_lng)
        all_lat = np.array(all_lat)
        
        lng_min, lng_max = np.min(all_lng), np.max(all_lng)
        lat_min, lat_max = np.min(all_lat), np.max(all_lat)
        self.geo_bounds = (lng_min, lng_max, lat_min, lat_max)
        
        print(f"经度范围：[{lng_min:.6f}, {lng_max:.6f}]")
        print(f"纬度范围：[{lat_min:.6f}, {lat_max:.6f}]")


    def generate_traj_npy(self) -> None:
        traj_length = self.config["traj_length"]
        traj_list = []
        
        for oid, group in tqdm(self.grouped_orders, desc="处理轨迹"):
            group_sorted = group.sort_values('GPS时间')
            lat = group_sorted['纬度'].values.astype(np.float64)
            lng = group_sorted['经度'].values.astype(np.float64)
            n_points = len(group_sorted)
            
            old_indices = np.arange(n_points)
            new_indices = np.linspace(0, n_points - 1, traj_length)
            interp_lat = interp1d(old_indices, lat, kind='linear', assume_sorted=True)(new_indices)
            interp_lng = interp1d(old_indices, lng, kind='linear', assume_sorted=True)(new_indices)
            
            traj = np.stack([interp_lat, interp_lng], axis=0).astype(np.float32)
            traj_list.append(traj)
        
        self.traj_array = np.stack(traj_list, axis=0)
        mean_lat = np.mean(self.traj_array[:, 0, :])
        std_lat = np.std(self.traj_array[:, 0, :]) + 1e-8
        mean_lng = np.mean(self.traj_array[:, 1, :])
        std_lng = np.std(self.traj_array[:, 1, :]) + 1e-8
        
        self.traj_array[:, 0, :] = (self.traj_array[:, 0, :] - mean_lat) / std_lat
        self.traj_array[:, 1, :] = (self.traj_array[:, 1, :] - mean_lng) / std_lng
        
        expected_shape = (len(self.grouped_orders), 2, traj_length)
        assert self.traj_array.shape == expected_shape, \
            f"traj形状错误！实际：{self.traj_array.shape}，预期：{expected_shape}"
        
        traj_path = os.path.join(self.output_dir, "traj.npy")
        np.save(traj_path, self.traj_array)
        self.traj_norm_params = {"mean_lat": mean_lat, "std_lat": std_lat, "mean_lng": mean_lng, "std_lng": std_lng}
        print(f"traj.npy 保存完成！路径：{os.path.abspath(traj_path)}，形状：{self.traj_array.shape}")


    def _get_grid_id(self, lng: float, lat: float) -> int:
        lng_min, lng_max, lat_min, lat_max = self.geo_bounds
        grid_size = self.config["grid_size"]
        
        grid_col = ((lng - lng_min) / (lng_max - lng_min + 1e-8)) * (grid_size - 1)
        grid_row = ((lat - lat_min) / (lat_max - lat_min + 1e-8)) * (grid_size - 1)
        
        grid_id = int(grid_row) * grid_size + int(grid_col)
        return np.clip(grid_id, 0, grid_size**2 - 1)


    def generate_head_npy(self) -> None:

        head_features = []
        for oid, group in tqdm(self.grouped_orders, desc="提取头部特征"):
            group_sorted = group.sort_values('GPS时间')
            coords = group_sorted[['经度', '纬度']].values
            start_time = group_sorted['GPS时间'].iloc[0]
            end_time = group_sorted['GPS时间'].iloc[-1]
            n_points = len(group_sorted)
            
            depature_idx = self.timestamp_to_depature_index(start_time)
            depature_idx = np.clip(depature_idx, 0, 287)
            
            total_distance = 0.0
            for i in range(1, n_points):
                total_distance += self.haversine_distance(
                    coords[i-1][0], coords[i-1][1],
                    coords[i][0], coords[i][1]
                )
            
            duration = (end_time - start_time).total_seconds()
            duration = max(duration, 1e-8)
            
            raw_length = n_points
            avg_step = total_distance / (raw_length - 1) if raw_length > 1 else 0.0
            avg_speed = total_distance / duration
            
            sid = self._get_grid_id(coords[0][0], coords[0][1])
            eid = self._get_grid_id(coords[-1][0], coords[-1][1])
            
            head_features.append([
                depature_idx, total_distance, duration, raw_length,
                avg_step, avg_speed, sid, eid
            ])
        
        self.head_array = np.array(head_features, dtype=np.float32)
        self.head_scaler = StandardScaler()
        self.head_array[:, 1:6] = self.head_scaler.fit_transform(self.head_array[:, 1:6])
        
        self.head_array[:, 0] = self.head_array[:, 0].astype(int)
        self.head_array[:, 6] = self.head_array[:, 6].astype(int)
        self.head_array[:, 7] = self.head_array[:, 7].astype(int)
        
        self._verify_head_features()
        
        head_path = os.path.join(self.output_dir, "head.npy")
        np.save(head_path, self.head_array)
        print(f"head.npy 保存完成！路径：{os.path.abspath(head_path)}，形状：{self.head_array.shape}")


    def _verify_head_features(self) -> None:
        depature_min, depature_max = self.head_array[:, 0].min(), self.head_array[:, 0].max()
        sid_min, sid_max = self.head_array[:, 6].min(), self.head_array[:, 6].max()
        eid_min, eid_max = self.head_array[:, 7].min(), self.head_array[:, 7].max()
        grid_max = self.config["grid_size"]**2 - 1
        
        print(f"出发时间索引范围：[{depature_min}, {depature_max}]（预期0~287）")
        print(f"起点ID（sid）范围：[{sid_min}, {sid_max}]（预期0~{grid_max}）")
        print(f"终点ID（eid）范围：[{eid_min}, {eid_max}]（预期0~{grid_max}）")
        
        expected_shape = (len(self.grouped_orders), 8)
    def save_norm_params(self) -> None:

        norm_params = {
            "traj": self.traj_norm_params,
            "head": {
                "mean": self.head_scaler.mean_,
                "std": self.head_scaler.scale_
            },
            "config": self.config
        }
        
        norm_path = os.path.join(self.output_dir, "norm_params.npy")
        np.save(norm_path, norm_params)


    def run(self) -> None:

        self.read_and_clean_data()
        self.group_and_filter_orders()
        self.calculate_geo_range()
        self.generate_traj_npy()
        self.generate_head_npy()
        self.save_norm_params()
        
        print("\n" + "="*60)
        print("处理完成  文件清单：")
        print(f"1. traj.npy：{os.path.abspath(os.path.join(self.output_dir, 'traj.npy'))}")
        print(f"2. head.npy：{os.path.abspath(os.path.join(self.output_dir, 'head.npy'))}")
        print(f"3. norm_params.npy：{os.path.abspath(os.path.join(self.output_dir, 'norm_params.npy'))}")
        print("="*60)
