import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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