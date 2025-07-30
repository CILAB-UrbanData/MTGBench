import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime as dt
from datetime import date, timedelta
import pandas as pd
from layers.Gcn_Related import graph_propagation_sparse_batch
from torch.nn import Parameter
from torch.nn import init
import math
from utils.tools import to_sparse_tensor, date_range
from utils.TrGNN import preprocess_data

# 1. 原始代码甚至没有实现data_provider train函数要整合一下, 超参数要整合一下 19621不要写死
# 2. 原始代码将数据预处理程序和模型代码混在一起，导致数据预处理和模型训练耦合过紧，结构不清晰

class ChannelFullyConnected(nn.Module):
        
    __constants__ = ['bias', 'in_features', 'channels']

    def __init__(self, in_features, channels, bias=True):
        super(ChannelFullyConnected, self).__init__()
        self.in_features = in_features
        self.channels = channels
        self.weight = Parameter(torch.Tensor(channels, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # input: (B, N, in_features)
        B, N, F = input.shape
        out = (input * self.weight.unsqueeze(0).unsqueeze(0)).sum(dim=2)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)
        return out  # (B, N)
    
    def extra_repr(self):
        return 'in_features={}, channels={}, bias={}'.format(
            self.in_features, self.channels, self.bias is not None
        )
    

class ChannelAttention(nn.Module):
        
    __constants__ = ['bias', 'in_features', 'channels']

    def __init__(self, in_features, out_features, channels, bias=True):
        super(ChannelAttention, self).__init__()
        self.in_features = in_features
        self.channels = channels
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(channels, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(channels, out_features)) # out_features=1+demand_hop
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # input: (B, H, N, in_features)
        B, H, N, F = input.shape
        # expand weight to batch
        w = self.weight.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, F, out)
        x = input.permute(0,2,1,3)  # (B, N, H, F)
        # compute: for each batch, channel, H
        # x: (B, N, H, F), w: (B, N, F, out)
        x_flat = x.reshape(B*N*H, F)
        w_flat = w.reshape(B*N, F, -1)
        y = torch.bmm(x_flat.unsqueeze(1), w_flat.repeat(H,1,1))  # (B*N*H,1,out)
        y = y.squeeze(1).reshape(B, N, H, -1).permute(0,2,1,3)  # (B,H,N,out)
        if self.bias is not None:
            y = y + self.bias.unsqueeze(0).unsqueeze(0)
        return y


    def extra_repr(self):
        return 'in_features={}, out_features={}, channels={}, bias={}'.format(
            self.in_features, self.out_features, self.channels, self.bias is not None
        )
    
    
class Model(nn.Module):
    # TrGNN.
    
    def __init__(self, input_size=1, output_size=1, demand_hop=75, status_hop=3):
        super(Model, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.demand_hop = demand_hop
        self.status_hop = status_hop
        
        # attention
        self.attention_layer = ChannelAttention(2**(status_hop+1)-1, demand_hop+1, channels=19621, bias=True) # channels=n_road
                
        # linear output
        self.output_layer = ChannelFullyConnected(in_features=4+24+1, channels=19621) # channels=n_road

        # other data stored here for convenience
        _, _, self.W, self.W_norm = preprocess_data()
               
    def forward(self, input, W=None, h_init=None):
        # X: graph signal. normalized. tensor: (history_window, n_road)
        # T: trajectory transition. normalized. tuple of history_window sparse_tensors: (n_road, n_road)
        # W: weighted road adjacency matrix. # sparse_tensor: (n_road, n_road)
        # h_init: for GRU. (gru_num_layers, n_road, hidden_size)
        # ToD: road-wise one-hot encoding of hour of day. (n_road, 24)
        # DoW: road-wise indicator. 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        X, T, ToD, DoW = input['x'], input['T'], input['ToD'], input['DoW']
        B, H, N = X.shape
        # demand propagation
        H_list=[]
        for t,A in enumerate(T):
            Ht = graph_propagation_sparse_batch(X[:,t,:], A.transpose(0,1), hop=self.demand_hop)
            H_list.append(Ht.unsqueeze(1))  # (B,1,N,hop+1)
        H = torch.cat(H_list, dim=1)  # (B,H,N,hop+1)
        # status propagation & attention
        S_list=[]
        for t in range(H):
            St = graph_propagation_sparse_batch(X[:,t,:], self.W_norm, hop=self.status_hop, dual=True)
            S_list.append(St.unsqueeze(1))
        S = torch.cat(S_list, dim=1)  # (B,H,N,1+2*status_hop)
        att = self.attention_layer(S.unsqueeze(4))
        att = F.softmax(att, dim=2)
        H = (H * att).sum(dim=3)  # (B,H,N)
        # combine features
        H = torch.cat([H.permute(0,2,1), ToD, DoW], dim=2)  # (B,N,H+25)
        # output
        Y = self.output_layer(H)  # (B,N)
        return Y