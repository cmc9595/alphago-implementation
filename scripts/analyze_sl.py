"""Post-training analysis of an SL checkpoint.

Evaluates every position of held-out games (the same split train_sl.py used), broken down
by data source and game phase, plus value-head calibration and quick sanity matches.

    python scripts/analyze_sl.py --model checkpoints/sl_6x96/final.pt
"""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from alphago import Board, replay_features
from alphago.data import GameData
from alphago.gtp import PolicyPlayer, load_net

PHASES = [(0, 30, "포석 1-30수"), (30, 80, "초반 31-80"), (80, 150, "중반 81-150"), (150, 1000, "후반 151+")]


@torch.no_grad()
def eval_games(net, data, games, device, batch=512):
    """Per-position (correct, top5, ply, value, z) for every move of the given games."""
    rows = []
    for g in games:
        start, end = int(data.offsets[g]), int(data.offsets[g + 1])
        n = end - start
        plies = np.arange(n, dtype=np.int32)
        for i in range(0, n, batch):
            pl = plies[i : i + batch]
            feats = replay_features(data.moves, np.full(len(pl), start, np.int64), pl)
            x = torch.from_numpy(feats).to(device).float()
            logits, value = net(x)
            target = torch.from_numpy(data.moves[start + pl].astype(np.int64)).to(device)
            top5 = logits.topk(5, 1).indices
            correct = (top5[:, 0] == target).cpu().numpy()
            in5 = (top5 == target[:, None]).any(1).cpu().numpy()
            v = value.cpu().numpy()
            z = data.results[g] * np.where(pl % 2 == 0, 1, -1)
            for k in range(len(pl)):
                rows.append((bool(correct[k]), bool(in5[k]), int(pl[k]), float(v[k]), int(z[k])))
    return rows


def summarize(rows):
    a = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    return {"positions": len(rows), "top1": float(a[0].mean()), "top5": float(a[1].mean())}


def play_match(black, white, games=10, seed=0):
    """Returns black's wins. Players are callables board -> action."""
    wins = 0
    for i in range(games):
        b = Board(19, 7.5)
        while not b.game_over and b.move_number < 450:
            b.play((black if b.to_move.name == "BLACK" else white)(b))
        wins += b.score() > 0
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sl_6x96/final.pt")
    ap.add_argument("--data", nargs="+", default=["data/kgs.npz", "data/fox_dan.npz"])
    ap.add_argument("--games-per-source", type=int, default=600)
    ap.add_argument("--out", default="docs/sl_report.json")
    args = ap.parse_args()

    net = load_net(args.model)
    device = next(net.parameters()).device
    data = GameData(args.data)  # identical split to training (same seed / order)
    n_kgs = len(np.load(args.data[0])["offsets"]) - 1
    val_games = data.splits["val"][0]
    rng = np.random.default_rng(0)
    report = {"model": args.model}

    by_source = {}
    all_rows = []
    for name, sel in (("KGS", val_games[val_games < n_kgs]), ("Fox 7-9d", val_games[val_games >= n_kgs])):
        pick = rng.choice(sel, min(args.games_per_source, len(sel)), replace=False)
        rows = eval_games(net, data, pick, device)
        by_source[name] = summarize(rows)
        all_rows += rows
        print(name, json.dumps(by_source[name]), flush=True)
    report["by_source"] = by_source
    report["overall"] = summarize(all_rows)

    report["by_phase"] = {}
    for lo, hi, label in PHASES:
        rows = [r for r in all_rows if lo <= r[2] < hi]
        report["by_phase"][label] = summarize(rows)

    # Value head: sign accuracy / MSE on positions with a usable outcome, by phase.
    report["value"] = {}
    for lo, hi, label in PHASES:
        rows = [r for r in all_rows if lo <= r[2] < hi and r[4] != 0]
        if rows:
            v = np.array([r[3] for r in rows])
            z = np.array([r[4] for r in rows])
            report["value"][label] = {"positions": len(rows), "winner_acc": float((np.sign(v) == z).mean()),
                                      "mse": float(((v - z) ** 2).mean()), "mean_abs_v": float(np.abs(v).mean())}

    # Sanity matches: greedy policy vs uniformly random sensible moves, and vs sampled policy.
    greedy = PolicyPlayer(net)
    rand_rng = np.random.default_rng(1)

    def random_move(b):
        legal = np.flatnonzero(b.sensible_mask()[:-1])
        return int(rand_rng.choice(legal)) if len(legal) else b.pass_action

    report["vs_random"] = {"games": 10, "policy_wins": play_match(greedy.genmove, random_move, 5)
                           + (5 - play_match(random_move, greedy.genmove, 5))}
    print(json.dumps(report, ensure_ascii=False, indent=1))
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
