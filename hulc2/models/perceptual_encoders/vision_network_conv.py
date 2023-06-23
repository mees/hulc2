#!/usr/bin/env python3

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class VisionNetworkConv(nn.Module):
    # reference: https://arxiv.org/pdf/2005.07648.pdf
    def __init__(
        self,
        activation_function: str,
        dropout_vis_fc: float,
        l2_normalize_output: bool,
        visual_features: int,
        num_c: int,
    ):
        super(VisionNetworkConv, self).__init__()
        self.l2_normalize_output = l2_normalize_output
        self.act_fn = getattr(nn, activation_function)()
        # model
        self.conv_model = nn.Sequential(
            # input shape: [N, 3, 200, 200]
            nn.Conv2d(in_channels=num_c, out_channels=32, kernel_size=8, stride=4),  # shape: [N, 32, 49, 49]
            nn.BatchNorm2d(32),
            self.act_fn,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),  # shape: [N, 64, 23, 23]
            nn.BatchNorm2d(64),
            self.act_fn,
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2),  # shape: [N, 64, 10, 10]
            nn.BatchNorm2d(64),
            self.act_fn,
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),  # shape: [N, 128, 4, 4]
            nn.BatchNorm2d(128),
            self.act_fn,
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1),  # shape: [N, 256, 1, 1]
            nn.BatchNorm2d(256),
            self.act_fn,
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            self.act_fn,
            nn.Dropout(dropout_vis_fc),
        )  # shape: [N, 512]
        self.fc2 = nn.Linear(in_features=512, out_features=visual_features)  # shape: [N, 64]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        if self.l2_normalize_output:
            x = F.normalize(x, p=2, dim=1)
        return x  # shape: [N, 64]
