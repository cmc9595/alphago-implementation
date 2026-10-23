"""Ownership / value labels from finished human games (KGS), double-checked against the record.

Keeps 19x19 even games that ended with two passes and a numeric result (e.g. RE[W+3.5]).
GNU Go judges dead stones; a game is kept only if GNU Go's score agrees with the score the
players recorded (within --tolerance points, same winner), so labels are confirmed by both.

    python scripts/prepare_human_ownership.py data/raw/kgs --out data/value/human_kgs.npz
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prepare_sgf import decode, iter_sgf_texts, validate  # noqa: E402

from alphago.gnugo import Referee, ownership  # noqa: E402
from alphago.sgf import parse_games  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=1.0)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--level", type=int, default=5, help="GNU Go level; 5 judged like 10 in our checks, ~2x faster")
    args = ap.parse_args()

    by_komi = defaultdict(list)
    for inp in args.inputs:
        for raw in iter_sgf_texts(inp):
            for g in parse_games(decode(raw)):
                m = re.match(r"([BW])\+(\d+(?:\.\d+)?)$", g.result_str.strip())
                if not m or g.komi is None or not g.moves or g.moves[-1][1] != 361:
                    continue
                v = validate(g, min_moves=100)
                if isinstance(v, str):
                    continue
                recorded = (1 if m.group(1) == "B" else -1) * float(m.group(2))
                by_komi[g.komi].append((v, recorded))
    total = sum(len(x) for x in by_komi.values())
    print(f"candidates: {total} games, komi values {sorted(by_komi)}", flush=True)

    rows, done, agree = [], 0, 0
    for komi, games in sorted(by_komi.items()):
        ref = Referee(args.procs, args.level, komi)
        try:
            for i in range(0, len(games), 500):
                chunk = games[i : i + 500]
                for (moves, recorded), v in zip(chunk, ref.judge_many([m for m, _ in chunk])):
                    done += 1
                    if v is None:
                        continue
                    score, dead = v
                    if np.sign(score) == np.sign(recorded) and abs(score - recorded) <= args.tolerance:
                        agree += 1
                        rows.append((moves, recorded, komi, ownership(moves, dead)))
                print(f"  judged {done}/{total}, kept {agree}", flush=True)
        finally:
            ref.close()

    lengths = np.array([len(r[0]) for r in rows])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        moves=np.concatenate([np.asarray(r[0], np.int16) for r in rows]),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        score=np.array([r[1] for r in rows], np.float32),  # recorded result, black - white (komi included)
        komi=np.array([r[2] for r in rows], np.float32),
        ownership=np.stack([r[3] for r in rows]),
    )
    print(f"kept {len(rows)}/{total} games ({len(rows) / max(total, 1):.0%}) -> {args.out}")


if __name__ == "__main__":
    main()
