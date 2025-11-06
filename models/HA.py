# ha_torch_batched.py
"""
读取历史上K天的历史进行平均，未完成dataset    
"""
from typing import Optional
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args, history_len: int):
        super().__init__()
        self.history_len = history_len
        self.timeslots_daynum = int(24 * 3600/args.time_slot)
        self.device = args.device
    
    def forward(self, input):
        # input: (B, D, T, N)  D为天数等于指定好的history_len, T为timeslots_daynum, N为节点数
        if input.dim() != 4:
            raise ValueError("Expected input with 4 dimensions (B, D, T, N), got {}".format(input.shape))
        
        B, D, T, N = input.shape
        assert D == self.history_len and T == self.timeslots_daynum, \
            f"Input shape mismatch: expected D={self.history_len}, T={self.timeslots_daynum}, got D={D}, T={T}"
        
        input = input.to(self.device)     
        # Compute the mean over the history dimension (D)
        return input.mean(dim=1)  # (B, T, N)