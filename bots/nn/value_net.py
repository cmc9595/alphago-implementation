# nn/value_net.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    """
    AlphaGo(2016) value network 스타일:
    - 입력: (B, 49, 19, 19)  # policy features 48 + current player color plane 1
    - conv(5x5) + 10~11x conv(3x3) + conv 추가 + 1x1 + FC256 + tanh
    """
    def __init__(self, in_planes: int = 49, channels: int = 192, board_size: int = 19):
        super().__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(in_planes, channels, kernel_size=5, padding=2)
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            for _ in range(10)  # 일단 10개; (정확 층수는 나중에 너가 맞추면 됨)
        ])
        self.conv_extra = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(channels, 1, kernel_size=1)

        self.fc1 = nn.Linear(board_size * board_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        for c in self.convs:
            x = F.relu(c(x))
        x = F.relu(self.conv_extra(x))
        x = self.conv_1x1(x)        # (B,1,19,19)
        x = x.flatten(1)            # (B,361)
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x)) # (B,1) in [-1,1]
        return x.squeeze(1)         # (B,)
