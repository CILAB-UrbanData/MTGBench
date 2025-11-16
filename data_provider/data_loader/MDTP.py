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
    def __init__(self, args, root_path, flag, normalization=True, S=24):
        super(MDTPSingleLoader, self).__init__()
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


class OtherForMDTP(Dataset):
    def __init__(self, args, root_path, flag, normalization=True, S=24):
        super(OtherForMDTP, self).__init__()
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
        N_all = len(data['X'])
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
            data['X'][idx_map[flag]], data['A'][idx_map[flag]], data['Y'][idx_map[flag]]
        )
        self.X_taxi_max = data['X_max']
        self.X_taxi_min = data['X_min']
        self.Y_taxi_max = data['Y_max']
        self.Y_taxi_min = data['Y_min']
    
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