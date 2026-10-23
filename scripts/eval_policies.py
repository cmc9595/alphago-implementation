"""Head-to-head evaluation of policy checkpoints (sampled play, GNU Go referee), as in RL training.

    python scripts/eval_policies.py A.pt B.pt --games 512
Prints A's win rate against B with a 95% interval; colours alternate.
"""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_rl import load_model  # noqa: E402

from alphago.gnugo import Referee  # noqa: E402
from alphago.models import pick_device  # noqa: E402
from alphago.selfplay import SelfPlayConfig, play_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    device = pick_device()
    a, _ = load_model(args.a, device)
    b, _ = load_model(args.b, device)
    a.eval()
    b.eval()
    cfg = SelfPlayConfig(temperature=args.temperature)
    ref = Referee(10, 5, cfg.komi)
    wins = n = 0
    try:
        for k in range(0, args.games, args.batch):
            recs = play_batch(a, b, min(args.batch, args.games - k), np.random.default_rng(args.seed + k),
                              device, cfg, referee=ref)
            wins += sum(r.learner_won for r in recs)
            n += len(recs)
    finally:
        ref.close()
    p = wins / n
    se = math.sqrt(p * (1 - p) / n)
    elo = 400 * math.log10(max(p, 1e-3) / max(1 - p, 1e-3))
    out = {"A": args.a, "B": args.b, "games": n, "A_win_rate": round(p, 3),
           "ci95": [round(p - 1.96 * se, 3), round(p + 1.96 * se, 3)], "elo_A_minus_B": round(elo)}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
