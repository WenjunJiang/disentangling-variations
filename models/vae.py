#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VAE(nn.Module):
    def Conv(self, nInput, nOutput, kernel_sz, stride_sz):
        if (stride_sz == 1) and (kernel_sz % 2 == 0):
            print('[-] Cannot have even kernel_sz without stride, increasing kernel_sz by 1')
            kernel_sz = kernel_sz + 1

        if stride_sz >= 1:
            padding = np.ceil((kernel_sz - stride_sz/2))
            return nn.Conv2d(nInput, nOutput, kernel_size=(kernel_sz, kernel_sz), \
                    stride=(stride_sz, stride_sz), padding=padding).to(device)

        else:
            padding = np.ceil((kernel_sz-1/stride_sz)/2)
            out_padding = (2 * padding) - (kernel_sz + 1 / stride_sz)
            return nn.ConvTranspose2d(nInput, nOutput, kernel_size=(kernel_sz, kernel_sz), \
                stride=(stride_sz, stride_sz), padding=padding, output_padding=out_padding).to(device)

    def ConvBlock(self, input, output, kernel_sz, stride_sz):
        self.conv1 = self.Conv(input, output, kernel_sz, stride_sz)
        self.batch_norm1 = nn.BatchNorm2d(output, momentum=0.1)
        self.relu1 = nn.ReLU()

    def __init__(self, config):
        super(VAE, self).__init__()

        self.config = config

        self.conv_block1 = self.ConvBlock()
        self.conv_block2 = self.ConvBlock()
        self.conv_block3 = self.ConvBlock()
        self.conv_block4 = self.ConvBlock()
        self.conv_block5 = self.ConvBlock()
        self.conv_block6 = self.ConvBlock()
        self.conv_block7 = self.ConvBlock()
        self.conv_block8 = self.ConvBlock()
        

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
