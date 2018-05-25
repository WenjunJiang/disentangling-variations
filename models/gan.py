#!/usr/bin/env python

import torch
import torch.nn as nn

class Descriminator(nn.Module):
    def __init__(self, config):
        super(Descriminator, self).__init__()
        self.config = config

    def forward(self, x):
        pass

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config

    def forward(self, x):
        pass
