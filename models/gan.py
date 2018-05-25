#!/usr/bin/env python

import torch
import torch.nn as nn

class LatentDiscriminator(nn.Module):
    def __init__(self, config):
        super(LatentDiscriminator, self).__init__()
        self.config = config

    def forward(self, x):
        pass

class PatchDiscriminator(nn.Module):
    def __init__(self, config):
        super(PatchDiscriminator, self).__init__()
        self.config = config

    def forward(self, x):
        pass        

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

    def forward(self, x):
        pass
