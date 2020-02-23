#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: model.py
# --- Creation Date: 24-02-2020
# --- Last Modified: Mon 24 Feb 2020 03:32:47 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Recognifion model
"""

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class VarPred(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(VarPred, self).__init__()
        self.conv0 = nn.Conv2d(in_channels,
                               64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = conv3x3(64, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = conv3x3(64, 128)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = conv3x3(128, 256)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.output = nn.Linear(256, out_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x1, x2 = torch.split(x, 2, dim=3)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.global_avgpool(x)
        x = self.output(x)

        return x
