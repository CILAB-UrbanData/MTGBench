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

class SF20_forTrGNN_Dataset(Dataset):  #TODO: 把sf删掉大部分低流量路段
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