import torch
# —— 初始化隐藏状态：全零  —— 
# 对 taxi 和 bike 两个分支分别做
def init_state(args):  # 返回 (h0, c0)
    num_layers = args.num_layers  # LSTM 层数
    B = args.batch_size * args.N_nodes  # B*N
    H = args.lstm_hidden
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # h,c: [num_layers, B*N, H]
    return (torch.zeros(num_layers, B, H, device=device),
            torch.zeros(num_layers, B, H, device=device))