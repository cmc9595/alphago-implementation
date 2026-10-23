"""GTP engine entry point for Sabaki and engine-vs-engine matches.

    python scripts/gtp_engine.py --model checkpoints/sl_6x96/last.pt

In Sabaki: Engines > Manage Engines > Add, path = <repo>/.venv/bin/python,
arguments = <repo>/scripts/gtp_engine.py --model <repo>/checkpoints/sl_6x96/last.pt
"""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

import argparse

from alphago.gtp import GTPEngine, PolicyPlayer, load_net
from alphago.mcts import MCTSPlayer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = always the top policy move")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--playouts", type=int, default=0, help="0 = raw policy; >0 = MCTS with this many playouts")
    ap.add_argument("--value-model", default=None, help="value net checkpoint (default: the model's own value head)")
    args = ap.parse_args()
    net = load_net(args.model)
    if args.playouts > 0:
        player = MCTSPlayer(net, load_net(args.value_model) if args.value_model else None, playouts=args.playouts)
    else:
        player = PolicyPlayer(net, temperature=args.temperature, seed=args.seed)
    GTPEngine(player).run()


if __name__ == "__main__":
    main()
