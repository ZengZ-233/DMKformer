import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2,Mamba
from timm.models.layers import trunc_normal_
from kan import KAN

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.mamba=Mamba(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            ).to("cuda")
        self.mamba2= Mamba2(
                # This module uses roughly 3 * expand * d_model^2 parameters
                d_model=d_model, # Model dimension d_model
                d_state=64,  # SSM state expansion factor, typically 64 or 128
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
            ).to("cuda")
        self.kan=KAN([d_model,96,d_model])

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)
        
        x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]
        y = x = [self.norm1(_x) for _x in x]
        # y = [self.dropout(self.activation(self.conv1(_y.transpose(-1, 1)))) for _y in y]
        # y = [self.dropout(self.conv2(_y).transpose(-1, 1)) for _y in y]
        # print("x",x[0].shape)
        y = [self.mamba(_y)  for _y in y]
        # y = [self.dropout(self.activation(self.mamba(_y))) for _y in y]
        # print("y",y[0].shape)
        # y = [self.mamba2(_y)  for _y in y]
        # z = [self.kan(_z)  for _z in z]

        # return [self.norm2(_x + _y +_z) for _x, _y,_z in zip(x, y, z)], attn

        return [self.norm2(_x + _y) for _x, _y in zip(x, y)], attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [[B, L1, D], [B, L2, D], ...]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        # concat all the outputs
        x = torch.cat(
            x, dim=1
        )  # (batch_size, patch_num_1 + patch_num_2 + ... , d_model)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
