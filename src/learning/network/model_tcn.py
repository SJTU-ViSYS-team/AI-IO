from typing import Optional
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import math


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
        self.linear1 = nn.Linear(num_channels[-1], output_size)
        self.linear2 = nn.Linear(num_channels[-1], output_size)
        self.rotor_spd_norm_layer = EmpiricalNormalization(4)
        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.01)
        self.linear2.weight.data.normal_(0, 0.01)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # x = x[:, :-6, :]
        start = 6
        rotor_spd = x[:, start:start+4, :]
        rotor_spd_squared = rotor_spd ** 2
        self.rotor_spd_norm_layer.update(rotor_spd_squared)
        rotor_spd_squared = self.rotor_spd_norm_layer(rotor_spd_squared)
        x[:, start:start+4, :] = rotor_spd_squared
        x = self.tcn(x)
        pred = self.linear1(x[:, :, -1])
        pred_cov = self.linear2(x[:, :, -1])
        return pred, pred_cov

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [T, 1, D]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class TransformerEncoderLayerWithAttention(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights: Optional[torch.Tensor] = None  # save attention weights

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        # original Self-Attention apply
        x_out, attn_weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )

        if not self.training:
            self.attn_weights = attn_weights.detach()

        return self.dropout1(x_out)

class IMUTransformerWithModality(nn.Module):
    def __init__(
        self,
        sub_dim=16,
        nhead=8,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2,
        output_size=3,
        window_size=100,
        enabled_modalities=["acc", "gyro", "rotor_spd"],  
    ):
        super(IMUTransformerWithModality, self).__init__()
        self.enabled_modalities = enabled_modalities
        self.modalities_cnn = nn.ModuleDict()

        num_enabled = len(self.enabled_modalities)
        d_model = num_enabled * sub_dim
        self.d_model = d_model
        if num_enabled == 0:
            raise ValueError("At least one modality must be enabled.")

        if "acc" in self.enabled_modalities:
            self.modalities_cnn["acc"] = nn.Conv1d(3, sub_dim, 5)
        if "gyro" in self.enabled_modalities:
            self.modalities_cnn["gyro"] = nn.Conv1d(3, sub_dim, 5)
        if "rotor_spd" in self.enabled_modalities:
            self.modalities_cnn["rotor_spd"] = nn.Conv1d(4, sub_dim, 5)

        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size)

        self.rotor_spd_norm_layer = EmpiricalNormalization(4)

        encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=False
            ) for _ in range(num_layers)
        ])
        self.transformer = nn.TransformerEncoder(encoder_layers[0], num_layers=1)  # dummy init
        self.transformer.layers = encoder_layers

        self.output_head = nn.Linear(d_model, output_size)
        self.output_cov = nn.Linear(d_model, output_size)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.output_head.weight, mean=0, std=0.01)
        nn.init.normal_(self.output_cov.weight, mean=0, std=0.01)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        """
        x: [B, C, T]  —  acc(3), gyro(3), rotor_spd(4), 6dR(6)
        """
        B, _, T = x.shape
        feat_list = []
        self.activations = {}

        if "acc" in self.enabled_modalities:
            acc = x[:, :3, :]
            acc_f = self.modalities_cnn["acc"](acc).permute(0, 2, 1)
            feat_list.append(acc_f)
            self.activations["acc"] = acc_f.detach().cpu()

        if "gyro" in self.enabled_modalities:
            gyro = x[:, 3:6, :]
            gyro_f = self.modalities_cnn["gyro"](gyro).permute(0, 2, 1)
            feat_list.append(gyro_f)
            self.activations["gyro"] = gyro_f.detach().cpu()

        if "rotor_spd" in self.enabled_modalities:
            start = 6
            rotor_spd = x[:, start:start+4, :]
            rotor_spd_squared = rotor_spd ** 2
            self.rotor_spd_norm_layer.update(rotor_spd_squared)
            rotor_spd_squared = self.rotor_spd_norm_layer(rotor_spd_squared)
            rotor_spd_f = self.modalities_cnn["rotor_spd"](rotor_spd_squared).permute(0, 2, 1)
            feat_list.append(rotor_spd_f)
            self.activations["rotor_spd"] = rotor_spd_f.detach().cpu()

        fused = torch.cat(feat_list, dim=-1)  # [B, T, C]
        fused = fused.permute(1, 0, 2)        # [T, B, C]

        x = self.pos_encoder(fused)
        x = self.transformer(x)
        last_step = x[-1, :, :]  # [B, C]
        pred = self.output_head(last_step)
        pred_cov = self.output_cov(last_step)
        return pred, pred_cov

class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, dim, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module."""
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(dim).unsqueeze(0))
        self.register_buffer("_var", torch.ones(dim).unsqueeze(0))
        self.register_buffer("_std", torch.ones(dim).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values."""
        assert x.dim() == 3, "Input must be a 3D tensor (B, C, T)."
        return (x - self._mean[..., None]) / (self._std[..., None] + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        assert x.dim() == 3, "Input must be a 3D tensor (B, C, T)."
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.size(-1))  # (B*T, C)

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        """De-normalize values based on empirical values."""

        return y * (self._std + self.eps) + self._mean
    