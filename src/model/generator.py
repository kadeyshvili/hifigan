import torch
import torch.nn as nn
from typing import List



class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        norm = nn.utils.parametrizations.weight_norm
        self.blocks = nn.ModuleList(nn.ModuleList([None for _ in range(len(dilations[0]))]) for _ in range(len(dilations)))
        for i in range(len(dilations)):
            for j in range(len(dilations[0])):
                self.blocks[i][j] = nn.Sequential(nn.LeakyReLU(0.1), norm(nn.Conv1d(in_channels=channels, out_channels=channels, \
                                                                                    kernel_size=kernel_size, dilation=dilations[i][j],padding="same")))
    
    def forward(self, x):
        for i in range(len(self.blocks)):
            res = x
            for j in range(len(self.blocks[i])):
                x = self.blocks[i][j](x)
            x = x + res
        return x
    
        

class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes, dilations):
        super().__init__()
        self.res_blocks = nn.ModuleList([ResBlock(channels=channels, kernel_size=kernel_sizes[i], dilations=dilations[i]) \
                                         for i in range(len(kernel_sizes))])

    def forward(self, x):
        result = self.res_blocks[0](x)
        for block in self.res_blocks[1:]:
            result += block(x)
        return result


class GenBlock(nn.Module):
    def __init__(self, in_channels, conv_trans_kernel_size, kernel_sizes_mrf, dilations):
        super().__init__()
        self.activation = nn.LeakyReLU(0.1)
        norm = nn.utils.parametrizations.weight_norm
        self.conv = norm(nn.ConvTranspose1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=conv_trans_kernel_size, \
                                            stride=conv_trans_kernel_size // 2, padding=(conv_trans_kernel_size - conv_trans_kernel_size // 2) // 2))
        self.mrf = MRF(channels=in_channels // 2, kernel_sizes=kernel_sizes_mrf, dilations=dilations )


    def forward(self, x):
        x = self.activation(x)
        x = self.conv(x)
        x = self.mrf(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_sizes_conv, kernel_sizes_mrf, dilations):
        super().__init__()
        norm = nn.utils.parametrizations.weight_norm
        num = len(kernel_sizes_conv)
        self.first_conv = norm(nn.Conv1d(in_channels=in_channels,out_channels=hidden_dim, kernel_size=7, dilation=1, padding="same"))
        self.blocks = nn.ModuleList([GenBlock(in_channels=hidden_dim // (2 ** i), conv_trans_kernel_size=kernel_sizes_conv[i], \
                                              kernel_sizes_mrf=kernel_sizes_mrf, dilations=dilations)for i in range(num)])
        last_conv_in_channels = hidden_dim // (2 ** num)
        self.relu = nn.LeakyReLU(0.1)
        self.last_conv = norm(nn.Conv1d(in_channels=last_conv_in_channels, out_channels=1, kernel_size=7, padding="same"))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.relu(x)
        x = self.last_conv(x)
        x = self.tanh(x)
        return x
