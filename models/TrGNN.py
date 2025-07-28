import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Gcn_Related import graph_propagation_sparse
from torch.nn import Parameter
from torch.nn import init
import math
from utils import to_sparse_tensor

# 1. 原始代码甚至没有实现data_provider train函数要整合一下, 超参数要整合一下 19621不要写死
# 2. 原始代码将数据预处理程序和模型代码混在一起，导致数据预处理和模型训练耦合过紧，结构不清晰

def normalize_adj(adj, mode='random walk'):
    # mode: 'random walk', 'aggregation'
    if mode == 'random walk': # for T. avg weight for sending node
        deg = np.sum(adj, axis=1).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(D_inv, adj)
    if mode == 'aggregation': # for W. avg weight for receiving node
        deg = np.sum(adj, axis=0).astype(np.float32)
        inv_deg = np.reciprocal(deg, out=np.zeros_like(deg), where=deg!=0)
        D_inv = np.diag(inv_deg)
        normalized_adj = np.matmul(adj, D_inv)
    return normalized_adj

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
        return torch.mul(input, self.weight).sum(dim=1) + self.bias
    
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

    def forward(self, input): # input: (history_window, channels=n_road, in_features=1+status_hop)
        return torch.mul(input, self.weight).sum(dim=2) + self.bias


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
        

    def forward(self, X, T, W_norm, ToD, DoW, W=None, h_init=None):
        # X: graph signal. normalized. tensor: (history_window, n_road)
        # T: trajectory transition. normalized. tuple of history_window sparse_tensors: (n_road, n_road)
        # W: weighted road adjacency matrix. # sparse_tensor: (n_road, n_road)
        # h_init: for GRU. (gru_num_layers, n_road, hidden_size)
        # ToD: road-wise one-hot encoding of hour of day. (n_road, 24)
        # DoW: road-wise indicator. 1 for weekdays, 0 for weekends/PHs. (n_road, 1)
        
        # graph propagation
        H = torch.cat([graph_propagation_sparse(x, A.transpose(0, 1), hop=self.demand_hop).unsqueeze(0) for x, A in zip(torch.unbind(X, dim=0), T)], dim=0)

        # attention
        S = torch.cat([graph_propagation_sparse(x, W_norm, hop=self.status_hop, dual=True).unsqueeze(0) for x in torch.unbind(X, dim=0)], dim=0)
        att = self.attention_layer(S.unsqueeze(3)) # specify weights and bias for each road segment
        att = F.softmax(att, dim=2) # attention weights across hops sum up to 1. (history_window, n_road, demand_hop+1)
        H = torch.mul(H, att) # (history_window, n_road, demand_hop+1)
        H = torch.sum(H, dim=2) # (history_window, n_road)
        
        # add ToD, DoW features
        H = torch.cat([H.transpose(0, 1), ToD, DoW], dim=1) # (n_road, history_window+24+1)
        
        # linear output. specify weights and bias for each road segment
        Y = self.output_layer(H) # (1, 1, n_road)

        return Y.squeeze(0).squeeze(0) 