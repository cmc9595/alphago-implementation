"""Policy / value networks: AlphaGo's roles, with a small modern ResNet trunk."""

from dataclasses import dataclass, replace

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
    global_pool: bool = False  # mean/max global-context bias in every third residual block


SCORE_SCALE = 20.0


class GlobalPoolBias(nn.Module):
    """KataGo-inspired (not identical): channelwise mean/max -> broadcast bias.

    Zero projection preserves a pretrained network at initialization while allowing
    the context path to learn immediately. No board-size scaling is needed for
    our fixed-size 19x19 networks.
    """
    def __init__(self, c: int):
        super().__init__()
        self.projection = nn.Linear(2 * c, c)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.projection(pooled)[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, c: int, global_pool=False):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.global_context = GlobalPoolBias(c) if global_pool else None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        context = self.global_context(y) if self.global_context is not None else None
        y = self.bn2(self.conv2(y))
        if context is not None:
            y = y + context
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
        self.trunk = nn.Sequential(*[ResBlock(c, global_pool=cfg.global_pool and i % 3 == 2)
                                     for i in range(cfg.blocks)])
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


def expand_blocks(net: GoNet, blocks: int, global_pool=None) -> GoNet:
    """Add identity blocks and/or zero global context, preserving initial outputs."""
    global_pool = net.cfg.global_pool if global_pool is None else global_pool
    if net.cfg.global_pool and not global_pool:
        raise ValueError('Cannot remove existing global context')
    adding_pool = global_pool and not net.cfg.global_pool
    if blocks < net.cfg.blocks or (blocks == net.cfg.blocks and not adding_pool):
        raise ValueError('Increase depth or add global pooling without reducing depth')
    if global_pool and blocks < 3:
        raise ValueError('Global pooling requires at least three blocks')
    expanded = GoNet(replace(net.cfg, blocks=blocks, global_pool=global_pool))
    state = expanded.state_dict()
    state.update(net.state_dict())
    expanded.load_state_dict(state, strict=True)
    with torch.no_grad():
        for block in expanded.trunk[net.cfg.blocks:]:
            block.bn2.weight.zero_()
            block.bn2.bias.zero_()
    # Trunk inputs are nonnegative, so ReLU(x + 0) is exactly x.
    return expanded.eval()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
