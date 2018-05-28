#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def Conv(self, in_channels, out_channels, kernel_sz, stride_sz):
        if (stride_sz == 1) and (kernel_sz % 2 == 0):
            print('[-] Cannot have even kernel_sz without stride, increasing kernel_sz by 1')
            kernel_sz = kernel_sz + 1

        if stride_sz >= 1:
            padding = np.ceil((kernel_sz - stride_sz/2))
            return nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_sz, kernel_sz), \
                    stride=(stride_sz, stride_sz), padding=padding).to(device)

        else:
            padding = np.ceil((kernel_sz-1/stride_sz)/2)
            out_padding = (2 * padding) - (kernel_sz + 1 / stride_sz)
            return nn.ConvTranspose2d(nInpuin_channelst, out_channels, kernel_size=(kernel_sz, kernel_sz), \
                stride=(stride_sz, stride_sz), padding=padding, output_padding=out_padding).to(device)

    def ConvBlock(self, in_channels, out_channels, kernel_sz, stride_sz):
        self.conv1 = self.Conv(in_channels, out_channels, kernel_sz, stride_sz)
        self.batch_norm1 = nn.BatchNorm2d(output, momentum=0.1)
        self.relu1 = nn.ReLU()

    def __init__(self, config):
        super(VAE, self).__init__()
        self.config = config

        self.conv_block_1 = self.ConvBlock(in_channels=3, out_channels=32, kernel_sz=5, stride_sz=2)
        self.conv_block_2 = self.ConvBlock(in_channels=32, out_channels=64, kernel_sz=5, stride_sz=2)
        self.conv_block_3 = self.ConvBlock(in_channels=64, out_channels=128, kernel_sz=5, stride_sz=2)
        self.conv_block_41 = self.ConvBlock(in_channels=128, out_channels=256, kernel_sz=5, stride_sz=2)
        self.conv_block_42 = self.ConvBlock(in_channels=128, out_channels=256, kernel_sz=5, stride_sz=2)
        self.conv_block_5 = self.ConvBlock(in_channels=256, out_channels=128, kernel_sz=5, stride_sz=1/2)
        self.conv_block_6 = self.ConvBlock(in_channels=128, out_channels=64, kernel_sz=5, stride_sz=1/2)
        self.conv_block_7 = self.ConvBlock(in_channels=64, out_channels=32, kernel_sz=5, stride_sz=1/2)
        self.conv_block_8 = self.ConvBlock(in_channels=32, out_channels=3, kernel_sz=5, stride_sz=1/2)

        # self.fc1 = nn.Linear(256, 256, bias=True)

    def encode(self, x):
        out = F.relu(self.conv_block_1(x))
        out = F.relu(self.conv_block_2(out))
        out = F.relu(self.conv_block_3(out))
        return self.conv_block_41(out), self.conv_block_42(out) 

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        out = F.relu(self.conv_block_5(z))
        out = F.relu(self.conv_block_6(out))
        out = F.relu(self.conv_block_7(out))
        return F.sigmoid(self.conv_block_8(out))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
