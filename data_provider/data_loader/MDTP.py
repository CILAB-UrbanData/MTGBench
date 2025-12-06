import os
import numpy as np
import pandas as pd
import torch
import pickle as pkl
import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

def load_trips(path, time_col, pu_col, do_col):
    """
    读取轨迹数据，确保包含：
      - 起始时间 time_col（datetime）
      - PULocationID pu_col（int）
      - DOLocationID do_col（int）
    """
    df = pd.read_parquet(path, engine="auto") if path.endswith('.parquet') else pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col])
    # 保留整点小时
    df['hour'] = df[time_col].dt.floor('H')
    df = df[[ 'hour', pu_col, do_col ]]
    return df

def build_time_index(df_taxi, df_bike):
    """合并两类数据，生成完整的时间序列索引（仅取交集）"""
    hours = pd.DatetimeIndex(
        sorted(set(df_taxi['hour']).intersection(df_bike['hour']))
    )
    return hours

def process_one_source(df, hours, loc_ids, pu_col, do_col):
    """
    对单一交通源（taxi 或 bike）：
    - X: [T, N, 2] 包括 in/out
    - A: [T, N, N] 区域间流量
    - Y: [T, N, 2] 下一小时的 in/out 标签
    """
    # 先只保留区域 ID 在 loc_ids 里的记录
    df = df[df[pu_col].isin(loc_ids) & df[do_col].isin(loc_ids)].copy()

    T = len(hours)
    N = len(loc_ids)
    id2idx = {loc: i for i, loc in enumerate(loc_ids)}

    # 初始化
    X = np.zeros((T, N, 2), dtype=np.float32)
    A = np.zeros((T, N, N), dtype=np.float32)
    Y = np.zeros((T, N, 2), dtype=np.float32)
    
    # 按小时分组
    grouped = df.groupby('hour')
    for t, hour in enumerate(tqdm(hours, desc='processing hours')):
        if hour in grouped.groups:
            sub = df.loc[grouped.groups[hour]]

            # -------- in/out per region --------
            # 先把原始 ID 映射到 [0, N) 的索引；映射不到会变成 NaN
            pu_mapped = sub[pu_col].map(id2idx)
            do_mapped = sub[do_col].map(id2idx)

            # 统计出度（out）和入度（in），并丢掉 NaN 索引
            pu_cnt = pu_mapped.value_counts()
            pu_cnt = pu_cnt[pu_cnt.index.notna()]
            do_cnt = do_mapped.value_counts()
            do_cnt = do_cnt[do_cnt.index.notna()]

            for loc_idx, cnt in pu_cnt.items():
                loc_idx = int(loc_idx)
                X[t, loc_idx, 0] = cnt   # out 流出到其他区域
            for loc_idx, cnt in do_cnt.items():
                loc_idx = int(loc_idx)
                X[t, loc_idx, 1] = cnt   # in 流入本区域

            # -------- 区域间流量 A[t, i, j] --------
            pairs = sub.groupby([pu_col, do_col]).size()
            for (pu, do), cnt in pairs.items():
                # 再保险：仅当两端都在 id2idx 时才写入
                if pu in id2idx and do in id2idx:
                    i, j = id2idx[pu], id2idx[do]
                    A[t, i, j] = cnt
    
    # 构造 Y：下一小时 in/out
    Y[:-1] = X[1:]
    # 最后一条记录保持 0 或者后续自己丢掉
    return X, A, Y

  
class MDTPRawloader(Dataset):
    def __init__(self, args, root_path, flag, S=24):
        super(MDTPRawloader, self).__init__()
        self.args = args
        self.root_path = root_path
        self.S = S
        self.data_path = "processed.npz"
        path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(path):
            # 1）加载原始数据
            df_taxi = load_trips(os.path.join(self.root_path, args.mdtp_taxi_path),  'tpep_pickup_datetime', 'PULocationID', 'DOLocationID')
            df_bike = load_trips(os.path.join(self.root_path, args.mdtp_bike_path),  'starttime',            'start region id', 'end region id')
            
            # 2）生成完整的时间轴（按小时）
            hours = build_time_index(df_taxi, df_bike)
            
            # 3）所有可能的区域 ID  正常切割肯定是切出比较整的区域数，但是可能很多区域没有数据，导致最大的区域id不等于我们切割的数量，
            # 因此这里读取的区域id数量比较奇怪，如果想统一可以从args里传入区域数量
            loc_ids = sorted(set(df_taxi['PULocationID']) 
                            | set(df_taxi['DOLocationID'])
                            | set(df_bike['start region id'])
                            | set(df_bike['end region id']))
            if self.args.data == 'NYCTAXI' or self.args.data == 'NYCFULL':
                loc_ids = list(range(264))  # NYC 固定区域数 264
            
            # 4）分别处理 taxi 和 bike
            print("=== Processing TAXI ===")
            X_taxi, A_taxi, Y_taxi = process_one_source(
                df_taxi, hours, loc_ids, 'PULocationID', 'DOLocationID'
            )
            print("=== Processing BIKE ===")
            X_bike, A_bike, Y_bike = process_one_source(
                df_bike, hours, loc_ids, 'start region id', 'end region id'
            )
            
            # 5）归一化（可选：根据全局最大最小）
            def minmax_norm(arr):
                mn, mx = arr.min(), arr.max()
                return (arr - mn) / (mx - mn + 1e-6), mn, mx
            X_taxi, X_taxi_min, X_taxi_max = minmax_norm(X_taxi)
            A_taxi, A_taxi_min, A_taxi_max = minmax_norm(A_taxi)
            Y_taxi, Y_taxi_min, Y_taxi_max = minmax_norm(Y_taxi)
            X_bike, X_bike_min, X_bike_max = minmax_norm(X_bike)
            A_bike, A_bike_min, A_bike_max = minmax_norm(A_bike)
            Y_bike, Y_bike_min, Y_bike_max = minmax_norm(Y_bike)
            
            # 6）保存到 npz
            np.savez_compressed(
                path,
                X_taxi=X_taxi,
                A_taxi=A_taxi,
                Y_taxi=Y_taxi,
                X_bike=X_bike,
                A_bike=A_bike,
                Y_bike=Y_bike,
                X_bike_max=X_bike_max,
                X_bike_min=X_bike_min,
                Y_bike_max=Y_bike_max,
                Y_bike_min=Y_bike_min,
                X_taxi_max=X_taxi_max,
                X_taxi_min=X_taxi_min,
                Y_taxi_max=Y_taxi_max,
                Y_taxi_min=Y_taxi_min,
                loc_ids=np.array(loc_ids),
                hours=hours.astype('datetime64[ns]')
            )
            print(f"Saved processed data to {path}")
        else:
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
        self.X_taxi_max = data['X_taxi_max']
        self.X_taxi_min = data['X_taxi_min']
        self.Y_taxi_max = data['Y_taxi_max']
        self.Y_taxi_min = data['Y_taxi_min']
        self.X_bike_max = data['X_bike_max']
        self.X_bike_min = data['X_bike_min']
        self.Y_bike_max = data['Y_bike_max']
        self.Y_bike_min = data['Y_bike_min']
    
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

class MDTPSingleLoader(Dataset):
    def __init__(self, args, root_path, flag, S=24):
        super(MDTPSingleLoader, self).__init__()
        self.args = args
        self.root_path = root_path
        self.S = S
        self.data_path = "processed.npz"
        path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(path):
            # 1）加载原始数据
            df_taxi = load_trips(os.path.join(self.root_path, args.mdtp_taxi_path),  'tpep_pickup_datetime', 'PULocationID', 'DOLocationID')     
            # 2）生成完整的时间轴（按小时）
            hours = build_time_index(df_taxi)
            
            # 3）所有可能的区域 ID
            loc_ids = sorted(set(df_taxi['PULocationID']) 
                            | set(df_taxi['DOLocationID']))
            if self.args.data == 'NYCTAXI' or self.args.data == 'NYCFULL':
                loc_ids = list(range(264))  # NYC 固定区域数 264
            
            # 4）分别处理 taxi 和 bike
            print("=== Processing TAXI ===")
            X_taxi, A_taxi, Y_taxi = process_one_source(
                df_taxi, hours, loc_ids, 'PULocationID', 'DOLocationID'
            )
            
            # 5）归一化（可选：根据全局最大最小）
            def minmax_norm(arr):
                mn, mx = arr.min(), arr.max()
                return (arr - mn) / (mx - mn + 1e-6), mn, mx
            X_taxi, X_taxi_min, X_taxi_max = minmax_norm(X_taxi)
            A_taxi, A_taxi_min, A_taxi_max = minmax_norm(A_taxi)
            Y_taxi, Y_taxi_min, Y_taxi_max = minmax_norm(Y_taxi)
            
            # 6）保存到 npz
            np.savez_compressed(
                path,
                X_taxi=X_taxi,
                A_taxi=A_taxi,
                Y_taxi=Y_taxi,
                X_taxi_max=X_taxi_max,
                X_taxi_min=X_taxi_min,
                Y_taxi_max=Y_taxi_max,
                Y_taxi_min=Y_taxi_min,
                loc_ids=np.array(loc_ids),
                hours=hours.astype('datetime64[ns]')
            )
            print(f"Saved processed data to {path}")
        else:
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
        self.X_taxi_max = data['X_taxi_max']
        self.X_taxi_min = data['X_taxi_min']
        self.Y_taxi_max = data['Y_taxi_max']
        self.Y_taxi_min = data['Y_taxi_min']
    
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
        y_taxi   = torch.stack(y_taxi)

        # 这里自动拼接 target
        target = y_taxi
        return (taxi_seq, A_taxi), target
