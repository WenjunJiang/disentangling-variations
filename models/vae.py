#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE(nn.Module):
    def Conv(self, in_channels, out_channels, kernel_sz, stride_sz):
        if (stride_sz == 1) and (kernel_sz % 2 == 0):
            kernel_sz = kernel_sz + 1

        if stride_sz >= 1:
            padding = np.ceil((kernel_sz - stride_sz)/2)
            return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sz, \
                    stride=stride_sz, padding=padding)

        else:
            padding = np.ceil((kernel_sz-(1/stride_sz))/2)
            out_padding = 2*padding - kernel_sz + 1/stride_sz
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_sz, \
                stride=1/stride_sz, padding=padding, output_padding=out_padding)

    def ConvBlock(self, in_channels, out_channels, kernel_sz, stride_sz):
        return nn.Sequential(
            self.Conv(in_channels, out_channels, kernel_sz, stride_sz),
            nn.BatchNorm2d(out_channels, momentum=self.config['momentum']),
            nn.ReLU()
        )

    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config

        self.conv_block_1 = self.ConvBlock(in_channels=3, out_channels=16, kernel_sz=3, stride_sz=1)
        self.conv_block_2 = self.ConvBlock(in_channels=16, out_channels=16, kernel_sz=3, stride_sz=1)
        self.conv_block_3 = self.ConvBlock(in_channels=16, out_channels=32, kernel_sz=2, stride_sz=2)
        self.conv_block_4 = self.ConvBlock(in_channels=32, out_channels=32, kernel_sz=3, stride_sz=1)
        self.conv_block_5 = self.ConvBlock(in_channels=32, out_channels=64, kernel_sz=2, stride_sz=2)
        self.conv_block_6 = self.ConvBlock(in_channels=64, out_channels=64, kernel_sz=3, stride_sz=1)
        self.conv_block_7 = self.ConvBlock(in_channels=64, out_channels=128, kernel_sz=2, stride_sz=2)
        self.conv_block_81 = self.Conv(in_channels=128, out_channels=128, kernel_sz=3, stride_sz=1)
        self.conv_block_82 = self.Conv(in_channels=128, out_channels=128, kernel_sz=3, stride_sz=1)

        self.conv_block_9 = self.ConvBlock(in_channels=128, out_channels=128, kernel_sz=3, stride_sz=1) 
        self.conv_block_10 = self.ConvBlock(in_channels=128, out_channels=64, kernel_sz=2, stride_sz=1/2)
        self.conv_block_11 = self.ConvBlock(in_channels=64, out_channels=64, kernel_sz=3, stride_sz=1)
        self.conv_block_12 = self.ConvBlock(in_channels=64, out_channels=32, kernel_sz=2, stride_sz=1/2)
        self.conv_block_13 = self.ConvBlock(in_channels=32, out_channels=32, kernel_sz=3, stride_sz=1)
        self.conv_block_14 = self.ConvBlock(in_channels=32, out_channels=16, kernel_sz=2, stride_sz=1/2)
        self.conv_block_15 = self.ConvBlock(in_channels=16, out_channels=16, kernel_sz=3, stride_sz=1)
        self.conv_block_16 = self.Conv     (in_channels=16, out_channels=3, kernel_sz=3, stride_sz=1)

        self.tanh_fun = nn.Tanh()
        # self.fc1 = nn.Linear(256, 256, bias=True)

    def encode(self, x):
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = self.conv_block_5(out)
        out = self.conv_block_6(out)
        out = self.conv_block_7(out)
        out1 = self.conv_block_81(out)
        out2 = self.conv_block_82(out)

        return out1, out2 

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        out = self.conv_block_9(z)
        out = self.conv_block_10(out)
        out = self.conv_block_11(out)
        out = self.conv_block_12(out)
        out = self.conv_block_13(out)
        out = self.conv_block_14(out)
        out = self.conv_block_15(out)
        out = self.conv_block_16(out)

        return self.tanh_fun(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
