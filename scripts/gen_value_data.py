"""Value-network training data, generated the AlphaGo (2016) way.

For each game: draw U, play moves 1..U-1 with the SL policy (diverse, human-like positions),
play move U uniformly at random among sensible moves, then let the RL policy play both sides
to the end (strong play -> accurate outcome). GNU Go judges the result and per-point ownership.
Only positions after U are labelled: s_{U+1} always, plus `--positions`-1 random later ones (AlphaGo kept one;
more than a few per game makes the value net memorise games).

GNU Go judges batch k on the CPU while the GPU plays batch k+1. Output is one .npz per batch
under --out; rerunning resumes after the last finished batch.

    python scripts/gen_value_data.py --games 250000
"""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_rl import load_model  # noqa: E402

from alphago import Board  # noqa: E402
from alphago.gnugo import Referee, ownership  # noqa: E402
from alphago.models import pick_device  # noqa: E402
from alphago.selfplay import policy_logits  # noqa: E402


def play_value_batch(sl, rl, n_games, rng, device, u_max, max_moves=400, komi=7.5):
    """Returns per game (moves, U). Games that end before move U are dropped by the caller."""
    n, pass_a = 19, 361
    boards = [Board(n, komi) for _ in range(n_games)]
    moves = [[] for _ in range(n_games)]
    us = rng.integers(1, u_max + 1, n_games)
    done = [False] * n_games
    while True:
        active = [i for i in range(n_games) if not done[i]]
        if not active:
            break
        groups = {"sl": [], "rand": [], "rl": []}
        for i in active:
            ply = len(moves[i]) + 1  # the move about to be played (1-based)
            groups["sl" if ply < us[i] else "rand" if ply == us[i] else "rl"].append(i)
        for kind, idxs in groups.items():
            if not idxs:
                continue
            if kind == "rand":
                choices = []
                for i in idxs:
                    legal = np.flatnonzero(boards[i].sensible_mask()[:pass_a])
                    choices.append(int(rng.choice(legal)) if len(legal) else pass_a)
            else:
                logits = policy_logits(sl if kind == "sl" else rl, [boards[i] for i in idxs], device)
                noise = rng.gumbel(size=logits.shape)  # sample from softmax(logits), T = 1
                choices = []
                for row, i in enumerate(idxs):
                    mask = boards[i].sensible_mask().astype(bool)
                    choices.append(int(np.argmax(np.where(mask, logits[row] + noise[row], -np.inf))))
            for i, a in zip(idxs, choices):
                boards[i].play(a)
                moves[i].append(a)
                if boards[i].game_over or len(moves[i]) >= max_moves:
                    done[i] = True
    return moves, us


def finish_batch(path, moves, us, referee, positions, rng):
    """Judge with GNU Go, pick labelled positions, write the batch file. Runs on a worker thread."""
    keep = [k for k in range(len(moves)) if len(moves[k]) > us[k]]  # game reached position U+1
    verdicts = referee.judge_many([moves[k] for k in keep])
    rows = []
    for k, v in zip(keep, verdicts):
        if v is None:
            continue
        score, dead = v
        u, m = int(us[k]), moves[k]
        # Candidate plies: positions after the random move (state after `ply` moves, ply >= U).
        # Always keep s_{U+1} (the position right after the random move, as in the AlphaGo paper);
        # the remaining positions-1 are drawn at random from the later plies.
        rest = np.arange(u + 1, len(m))
        extra = rng.choice(rest, min(positions - 1, len(rest)), replace=False) if positions > 1 else []
        plies = np.sort(np.concatenate([[u], extra]).astype(np.int64))
        rows.append((m, u, plies, score, ownership(m, dead)))
    if not rows:
        return 0
    lengths = np.array([len(r[0]) for r in rows])
    plies = np.full((len(rows), positions), -1, np.int32)
    for j, r in enumerate(rows):
        plies[j, : len(r[2])] = r[2]
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        moves=np.concatenate([np.asarray(r[0], np.int16) for r in rows]),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        u=np.array([r[1] for r in rows], np.int32),
        plies=plies,  # -1 = unused slot
        score=np.array([r[3] for r in rows], np.float32),  # black - white, komi included
        ownership=np.stack([r[4] for r in rows]),  # +1 black / -1 white / 0 neutral
    )
    tmp.replace(path)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", default="checkpoints/sl_6x96/final.pt")
    ap.add_argument("--rl", default="checkpoints/rl_6x96/final.pt")
    ap.add_argument("--out", default="data/value/gen")
    ap.add_argument("--games", type=int, default=250_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--u-max", type=int, default=280, help="U ~ uniform{1..u_max} (games average ~310 moves)")
    ap.add_argument("--positions", type=int, default=3)
    ap.add_argument("--referee-procs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    done_batches = sorted(out.glob("batch_*.npz"))
    start = len(done_batches)
    total_batches = (args.games + args.batch - 1) // args.batch
    device = pick_device()
    sl, _ = load_model(args.sl, device)
    rl, _ = load_model(args.rl, device)
    sl.eval()
    rl.eval()
    referee = Referee(args.referee_procs, 5)
    ex = ThreadPoolExecutor(1)
    log = out / "log.jsonl"
    print(f"resuming at batch {start}/{total_batches}", flush=True)

    pending = None
    t_last = time.time()
    for b in range(start, total_batches):
        rng = np.random.default_rng(args.seed * 1_000_003 + b)  # deterministic per batch
        t0 = time.time()
        moves, us = play_value_batch(sl, rl, args.batch, rng, device, args.u_max)
        t_play = time.time() - t0
        if pending is not None:
            kept = pending[1].result()
            rec = {"batch": pending[0], "games_kept": kept, "sec_play": round(t_play, 1),
                   "sec_total": round(time.time() - t_last, 1)}
            t_last = time.time()
            print(json.dumps(rec), flush=True)
            with open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
        pending = (b, ex.submit(finish_batch, out / f"batch_{b:05d}.npz", moves, us, referee,
                                args.positions, np.random.default_rng(rng.integers(1 << 62))))
    if pending is not None:
        print(json.dumps({"batch": pending[0], "games_kept": pending[1].result()}), flush=True)
    ex.shutdown()
    referee.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()
