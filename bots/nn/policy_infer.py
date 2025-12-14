# bots/nn/policy_infer.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

from bots.nn.policy_net import PolicyNet
from go.board import Board, Move
from features.alpha48 import FeatureState, extract_alpha48_planes

class PolicyPrior:
    def __init__(self, weight_path: Path, device: str = "cpu"):
        self.device = device
        self.model = PolicyNet().to(device)
        sd = torch.load(weight_path, map_location=device)
        self.model.load_state_dict(sd)
        self.model.eval()

    @torch.no_grad()
    def priors(self, board: Board, st: FeatureState) -> dict[Move, float]:
        """
        return: {move: prior_prob} only for legal moves (incl pass optionally)
        """
        x = extract_alpha48_planes(board, st, include_value_color_plane=False)  # (48,19,19)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1,48,19,19)

        logits = self.model(x)[0]  # (361 or 362)
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()  # (361/362,)

        pri: dict[Move, float] = {}
        for m in board.legal_moves():
            if m.is_resign:
                continue
            if m.is_pass:
                idx = board.size * board.size
            else:
                idx = m.y * board.size + m.x
            pri[m] = float(probs[idx])

        # normalize (safe)
        s = sum(pri.values())
        if s > 0:
            for k in list(pri.keys()):
                pri[k] /= s
        return pri
