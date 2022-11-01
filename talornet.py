from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os
from os import path
from os.path import join as pjoin
from matplotlib import image
from matplotlib.pyplot import cla

import torch
import torch.nn as nn
import numpy as np
from models import f_net, Unet
BatchNorm2d = nn.BatchNorm2d

import torchvision


class TL_net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.f_model = f_net(inchannels=3, interchannels=64, outchannels=3)
        self.g_model = UNet(in_channels=3, num_classes=3, init_features=32)
        self.tl_layer_num = 3

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        f_x = self.f_model(x)
        features = []
        features.append(f_x)
        coeffici = []
        base_co = 1
        coeffici.append(base_co)
        for i in range(self.tl_layer_num):
            base_co *= (i + 1)
            g_in = torch.cat([x, features[-1]], dim=1)
            i_g_feature = self.g_model(g_in)
            i_g_feature = i_g_feature + i * features[-1]
            features.append(i_g_feature)
            coeffici.append(base_co)

        tl_out = torch.zeros_like(f_x)
        for feature, co in zip(features, coeffici):
            tl_out += feature / co

        return tl_out