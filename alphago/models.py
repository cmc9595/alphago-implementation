"""Policy / value networks: AlphaGo's roles, with a small modern ResNet trunk."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from alphago import NUM_FEATURES


@dataclass
class NetConfig:
    board_size: int = 19
    blocks: int = 6
    channels: int = 96
    in_planes: int = NUM_FEATURES
    policy: bool = True
    value: bool = True
    ownership: bool = False  # KataGo-style auxiliary head: final owner of every point
    score: bool = False  # KataGo-style auxiliary head: final score margin (side to move), in units of 20 points


SCORE_SCALE = 20.0


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class GoNet(nn.Module):
    """Shared trunk with optional policy head (N*N+1 logits incl. pass) and value head (tanh)."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        n, c = cfg.board_size, cfg.channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_planes, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU()
        )
        self.trunk = nn.Sequential(*[ResBlock(c) for _ in range(cfg.blocks)])
        if cfg.policy:
            self.policy_conv = nn.Sequential(nn.Conv2d(c, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU())
            self.policy_fc = nn.Linear(2 * n * n, n * n + 1)
        if cfg.value:
            self.value_conv = nn.Sequential(nn.Conv2d(c, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
            self.value_fc = nn.Sequential(nn.Linear(n * n, 256), nn.ReLU(), nn.Linear(256, 1))
        if cfg.ownership:
            self.ownership_conv = nn.Conv2d(c, 1, 1)
        if cfg.score:
            self.score_conv = nn.Sequential(nn.Conv2d(c, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU())
            self.score_fc = nn.Sequential(nn.Linear(n * n, 128), nn.ReLU(), nn.Linear(128, 1))

    def ownership(self, x):
        """Per-point final ownership in [-1, 1] from the side to move's view, shape (B, N, N)."""
        h = self.trunk(self.stem(x))
        return torch.tanh(self.ownership_conv(h)).squeeze(1)

    def forward_all(self, x):
        """(policy | None, value | None, ownership | None) from one trunk pass."""
        h = self.trunk(self.stem(x))
        policy = self.policy_fc(self.policy_conv(h).flatten(1)) if self.cfg.policy else None
        value = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(1) if self.cfg.value else None
        own = torch.tanh(self.ownership_conv(h)).squeeze(1) if self.cfg.ownership else None
        return policy, value, own

    def forward_heads(self, x):
        """(policy, value, ownership, score) from one trunk pass; missing heads are None.
        score is unbounded, side-to-move view, in units of SCORE_SCALE points."""
        h = self.trunk(self.stem(x))
        policy = self.policy_fc(self.policy_conv(h).flatten(1)) if self.cfg.policy else None
        value = torch.tanh(self.value_fc(self.value_conv(h).flatten(1))).squeeze(1) if self.cfg.value else None
        own = torch.tanh(self.ownership_conv(h)).squeeze(1) if self.cfg.ownership else None
        score = self.score_fc(self.score_conv(h).flatten(1)).squeeze(1) if self.cfg.score else None
        return policy, value, own, score

    def forward(self, x, detach_value_trunk=False):
        """x: float tensor (B, planes, N, N). Returns (policy_logits | None, value | None).

        detach_value_trunk: the value head reads the trunk but its loss does not train it
        (RL uses the value head only as a baseline and must not let it reshape policy features).
        """
        h = self.trunk(self.stem(x))
        policy = value = None
        if self.cfg.policy:
            policy = self.policy_fc(self.policy_conv(h).flatten(1))
        if self.cfg.value:
            hv = h.detach() if detach_value_trunk else h
            value = torch.tanh(self.value_fc(self.value_conv(hv).flatten(1))).squeeze(1)
        return policy, value


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
