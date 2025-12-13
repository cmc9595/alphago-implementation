# bots/nn/policy_net.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    AlphaGo-style Supervised Policy Network
    Input:  (B, 48, 19, 19)
    Output: (B, 361)
    """

    def __init__(self, board_size: int = 19, in_planes: int = 48, channels: int = 192, num_blocks: int = 12):
        super().__init__()
        self.board_size = board_size

        layers = []

        # First layer: 5x5
        layers.append(nn.Conv2d(in_planes, channels, kernel_size=5, padding=2))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: 3x3
        for _ in range(num_blocks):
            layers.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        self.trunk = nn.Sequential(*layers)

        # Policy head: 1x1 conv
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 48, 19, 19)
        returns logits: (B, 361)
        """
        h = self.trunk(x)
        p = self.policy_head(h)          # (B,1,19,19)
        p = p.view(p.size(0), -1)         # (B,361)
        return p
