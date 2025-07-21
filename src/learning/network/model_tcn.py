"""
This file is part of Learned Inertial Model Odometry.
Copyright (C) 2023 Giovanni Cioffi <cioffi at ifi dot uzh dot ch>
(Robotics and Perception Group, University of Zurich, Switzerland).
This file is subject to the terms and conditions defined in the file
'LICENSE', which is part of this source code package.
"""

"""
Reference: https://github.com/CathIAS/TLIO/blob/master/src/network/model_tcn.py
"""

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn.utils import weight_norm


dict_activation = {"ReLU": nn.ReLU, "GELU": nn.GELU}


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
        activation=nn.ReLU,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.activation1 = activation()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.activation2 = activation()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.activation1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.activation2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_hidden_channels,
        kernel_size=2,
        dropout=0.2,
        activation="ReLU",
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_hidden_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_hidden_channels[i - 1]
            out_channels = num_hidden_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    activation=activation,
                )
            ]

        # print("receptive field = ", 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # only return last component
        return self.network(x)


class Tcn(nn.Module):
    """
    This tcn is trained so that the output at current time is a vector that contains
    the prediction using the last second of inputs.
    The receptive field is givent by the input parameters.
    """

    def __init__(
        self,
        input_size,
        output_size,
        num_channels,
        kernel_size,
        dropout,
        activation="ReLU",
    ):
        super(Tcn, self).__init__()
        self.tcn = TemporalConvNet(
            input_size,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=dict_activation[activation],
        )
        # self.gru = nn.GRU(input_size = 128, hidden_size =64, num_layers = 1, batch_first = True,bidirectional=True)
        self.linear1 = nn.Linear(num_channels[-1], output_size)
        self.linear2 = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.01)
        self.linear2.weight.data.normal_(0, 0.01)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.tcn(x)
        # x = x.permute(0, 2, 1)
        # x, _ = self.gru(x)
        # x = x.permute(0, 2, 1)
        pred = self.linear1(x[:, :, -1])
        pred_cov = self.linear2(x[:, :, -1])
        return pred, pred_cov

class LearnableFilter(nn.Module):
    def __init__(self, channels=3, kernel_size=5, init_as_mean=True):
        """
        :param channels: 输入通道数（对应加速度计三轴）
        :param kernel_size: 卷积核大小（决定滤波窗口长度）
        :param init_as_mean: 是否初始化为均值滤波器
        """
        super().__init__()
        self.kernel_size = kernel_size
        
        # 使用深度可分离卷积（Depthwise Conv1D）
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,  # 输出通道与输入相同
            kernel_size=kernel_size,
            padding= (kernel_size//2),  # 保持时序长度不变
            groups=channels,  # 关键！每个通道独立卷积
            bias=False  # 滤波器不需要偏置
        )
        
        # 初始化策略
        if init_as_mean:
            # 初始化为均值滤波器（类似滑动平均）
            self.conv.weight.data = torch.ones_like(self.conv.weight) / kernel_size
            self.conv.weight.requires_grad = True  # 允许后续学习调整
    
    def forward(self, x):
        """
        :param x: 输入加速度数据 [Batch, Channels, Time Steps]
        :return: 滤波后的数据 [Batch, Channels, Time Steps]
        """
        return self.conv(x)

class NoiseAwareTCN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_channels,
        kernel_size,
        dropout,
        activation="ReLU",
    ):
        super().__init__()
        self.filter = LearnableFilter(channels=3, kernel_size=5)
        self.tcn = TemporalConvNet(
            input_size,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=dict_activation[activation],
        )
        self.linear1 = nn.Linear(num_channels[-1], output_size)
        self.linear2 = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.01)
        self.linear2.weight.data.normal_(0, 0.01)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        angles = x[:, :3]
        acc = x[:, 3:6]
        filtered_acc = self.filter(acc)
        cleaned_input = torch.cat([angles, filtered_acc], dim=1)
        x = self.tcn(cleaned_input)
        pred = self.linear1(x[:, :, -1])
        pred_cov = self.linear2(x[:, :, -1])
        return pred, pred_cov
    
class TcnWithAttention(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_channels,
        kernel_size,
        dropout,
        activation="ReLU",
        attn_heads=4
    ):
        super().__init__()
        
        # 原始TCN部分保持不变
        self.tcn = TemporalConvNet(
            input_size,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=dict_activation[activation],
        )
        
        # 新增注意力机制
        self.attention = MultiheadAttention(
            embed_dim=num_channels[-1],  # 输入维度与TCN输出通道一致
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=False  # 适配TCN的输出维度
        )
        
        # 新增注意力后的特征融合层
        self.attn_proj = nn.Linear(num_channels[-1], num_channels[-1])
        
        # 修改原始全连接层（保持双输出结构）
        self.linear1 = nn.Linear(num_channels[-1] * 2, output_size)  # 拼接原始和注意力特征
        self.linear2 = nn.Linear(num_channels[-1] * 2, output_size)
        
        self.init_weights()

    def init_weights(self):
        # 原始初始化
        self.linear1.weight.data.normal_(0, 0.01)
        self.linear2.weight.data.normal_(0, 0.01)
        # 注意力相关初始化
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.constant_(self.attention.out_proj.bias, 0.1)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # 原始TCN处理 [batch, channels, time]
        tcn_out = self.tcn(x)  # 假设输入x形状为 [B, input_size, T]
        
        # 维度转换以适应注意力机制
        # [B, C, T] → [T, B, C] （时间步作为序列长度）
        tcn_out_perm = tcn_out.permute(2, 0, 1)  
        
        # 自注意力处理
        attn_out, _ = self.attention(
            query=tcn_out_perm,
            key=tcn_out_perm,
            value=tcn_out_perm
        )  # 输出形状 [T, B, C]
        
        # 特征融合（残差连接+投影）
        fused_feature = self.attn_proj(attn_out + tcn_out_perm)  # [T, B, C]
        
        # 拼接原始TCN特征和注意力特征（沿通道维度）
        final_feature = torch.cat([
            tcn_out[:, :, -1],          # 原始TCN最后时间步 [B, C]
            fused_feature[-1, :, :]     # 注意力最后时间步 [B, C]
        ], dim=1)  # → [B, 2C]
        
        # 双输出结构保持不变
        pred = self.linear1(final_feature)
        pred_cov = self.linear2(final_feature)
        return pred, pred_cov