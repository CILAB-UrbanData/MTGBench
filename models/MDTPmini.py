import torch
import torch.nn as nn
from layers.Gcn_Related import gcn_norm

class GraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
    def forward(self, A, X):
        # A: [batch, N, N], X: [batch, N, F]
        A_norm = gcn_norm(A)
        out = A_norm @ X  # [batch, N, F]
        return self.linear(out)

class Model(nn.Module):
    def __init__(self, args):
        #in_feats=2,gcn_hidden=128,lstm_hidden=256,fusion='sum',dropout=0.5
        super().__init__()
        # 两个分支：Taxi & Bike
        self.gcn_taxi = GraphConv(args.in_feats, args.gcn_hidden)  #in_feats包含入度和出度, gcn_hidden参考原文设置
        self.gcn_bike = GraphConv(args.in_feats, args.gcn_hidden)
        # 双层 LSTM，用于捕捉时间依赖
        self.lstm_taxi = nn.LSTM(args.gcn_hidden, args.lstm_hidden, num_layers=2,
                                batch_first=True, dropout=args.dropout)  
        self.lstm_bike = nn.LSTM(args.gcn_hidden, args.lstm_hidden, num_layers=2,
                                batch_first=True, dropout=args.dropout)
        fusion_dim = args.lstm_hidden
        # 融合后两层全连接
        self.fc1 = nn.Linear(fusion_dim, fusion_dim)
        # 第二层接受 fc1 输出拼接融合特征
        self.fc2 = nn.Linear(fusion_dim, 2) #输出为预测的taxi的in out
        # 权重初始化
        self.apply(self._init_weights)
        # 隐变量初始化
        self.state_taxi = None
        self.args = args
    
    def reset_state(self):
        self.state_taxi = None

    def _init_state(self, batch_size):
        h = torch.zeros(2, batch_size * self.args.N_nodes, self.args.lstm_hidden).to(self.args.device) #2表示有两层隐变量，2层是按照原论文来的
        c = torch.zeros(2, batch_size * self.args.N_nodes, self.args.lstm_hidden).to(self.args.device)
        return (h, c)

    def _detach_state(self, state):
        if state is None:
            return None
        h, c = state
        return (h.detach(), c.detach())

    def _init_weights(self, module):
        """
        对模型中可学习的层进行初始化。
        - Linear 层：Xavier 正态分布初始化权重，偏置置零
        - LSTM：使用 Kaiming 正态对输入-隐藏权重，正态对隐藏-隐藏权重，偏置置零
        - GCN 中的 Linear 同 Linear 层处理
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
        # GraphConv 中的 Linear 也会被上述 Linear 分支初始化
    
    def forward_branch(self, seq_X, seq_A, gcn, lstm, state):
        B, S, N, F = seq_X.shape
        # 1) 图卷积：对每个时间步 t 依次聚合空间特征
        gx = []
        for t in range(S):
            xt = seq_X[:, t]      # [B, N, F]
            At = seq_A[:, t]      # [B, N, N]
            out_t = torch.relu(gcn(At, xt))  # [B, N, gcn_hidden]
            gx.append(out_t)
        # 堆叠成 [B, S, N, gcn_hidden]
        gx = torch.stack(gx, dim=1)

        # 2) 为了送进 LSTM，需把 (B, N) 两维“展平”成一组序列
        #    先调到 [B, N, S, H]，再 reshape 到 [B*N, S, H]
        gx = gx.permute(0, 2, 1, 3).reshape(B * N, S, -1)

        # 3) LSTM 时间建模：取最后一层、最后时刻的隐藏态 h_n[-1]
        out, new_state = lstm(gx, state)
        h_n, c_n = new_state                     # each [num_layers, B*N, hidden]
        # 取最后一层最后时刻
        h_last = h_n[-1].reshape(B, N, -1)       # [B,N,hidden]
        return h_last, new_state
    
    def forward(self, input):
        taxi_seq, A_taxi = input

        taxi_seq = taxi_seq.to(self.args.device)  # [B, S, N
        A_taxi = A_taxi.to(self.args.device)

        B = self.args.batch_size

        if self.state_taxi is None:
            self.state_taxi = self._init_state(batch_size=B)
        # 分别得出租车 & 单车分支输出
        h_taxi, self.state_taxi = self.forward_branch(taxi_seq, A_taxi, self.gcn_taxi, self.lstm_taxi, self.state_taxi)
        
        H = h_taxi
        # 两层全连接预测
        x1 = torch.relu(self.fc1(H))                           # [B, N, fusion_dim]
        pred = self.fc2(x1)                                    # [B, N, 2]
        self.state_taxi = self._detach_state(self.state_taxi)
        return pred # 每个节点的 [in_pred, out_pred]