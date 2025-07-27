import torch
import torch.nn as nn
import torch.nn.functional as F

# 实现膨胀因果卷积（Dilated Causal Convolution）
class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # padding = dilation*(kernel_size-1) 实现因果 (causal) 卷积
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )

    def forward(self, x):
        # x: [N, C_in, T_in]
        x = F.pad(x, (self.pad, 0))     # 只在时间维度左侧补 pad
        return self.conv(x)            # 输出 [N, C_out, T_out]