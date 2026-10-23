"""Policy network vs GNU Go, alternating colours, several games in parallel.

    python scripts/match_gnugo.py --model checkpoints/sl_6x96/final.pt --games 40 --level 10

GNU Go's final_score (which removes dead stones) decides each game. SGFs go to runs/match_*/.
"""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

import argparse
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alphago import Board, Color
from alphago.gnugo import GnuGo
from alphago.gtp import PolicyPlayer, action_to_vertex, load_net, vertex_to_action


def sgf_point(a):
    if a == 361:
        return ""
    x, y = a % 19, a // 19
    return chr(97 + x) + chr(97 + 18 - y)


def play_game(idx, ours_black, player, net_lock, args, out_dir):
    g = GnuGo(args.level, args.komi, superko=True)
    b = Board(19, args.komi, superko=True)
    moves = []
    try:
        g.cmd("boardsize 19")
        g.cmd("clear_board")
        resigned = False
        while not b.game_over and b.move_number < args.max_moves:
            black = b.to_move == Color.BLACK
            if black == ours_black:
                with net_lock:
                    a = player.genmove(b)
                g.cmd(f"play {'B' if black else 'W'} {action_to_vertex(a, 19)}")
            else:
                v = g.cmd(f"genmove {'B' if black else 'W'}")
                if v.lower() == "resign":
                    resigned, winner_black = True, not black
                    break
                a = vertex_to_action(v, 19)
            b.play(a)
            moves.append((a, black))
        if resigned:
            result = "B+R" if winner_black else "W+R"
        else:
            result = g.cmd("final_score")  # GNU Go removes dead stones: "B+12.5" / "W+3.5"
            winner_black = result.upper().startswith("B+")
    finally:
        g.close()
    ours_won = winner_black == ours_black
    body = "".join(f";{'B' if blk else 'W'}[{sgf_point(a)}]" for a, blk in moves)
    pb, pw = ("policy", f"gnugo-{args.level}") if ours_black else (f"gnugo-{args.level}", "policy")
    (out_dir / f"game{idx:03d}.sgf").write_text(
        f"(;GM[1]FF[4]SZ[19]KM[{args.komi}]RU[Chinese]PB[{pb}]PW[{pw}]RE[{result}]{body})")
    return {"game": idx, "ours_black": ours_black, "result": result, "ours_won": ours_won, "moves": len(moves)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sl_6x96/final.pt")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--level", type=int, default=10, help="GNU Go level 1-10 (10 = strongest default)")
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--max-moves", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    out_dir = Path("runs") / f"match_{Path(args.model).parent.name}_{Path(args.model).stem}_gnugo{args.level}"
    out_dir.mkdir(parents=True, exist_ok=True)
    player = PolicyPlayer(load_net(args.model), temperature=args.temperature)
    net_lock = threading.Lock()
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(args.parallel) as ex:
        futs = [ex.submit(play_game, i, i % 2 == 0, player, net_lock, args, out_dir) for i in range(args.games)]
        for f in futs:
            r = f.result()
            results.append(r)
            wins = sum(x["ours_won"] for x in results)
            print(f"game {r['game']:3d} ours={'B' if r['ours_black'] else 'W'} {r['result']:>8} "
                  f"{'WIN ' if r['ours_won'] else 'loss'}  running {wins}/{len(results)}", flush=True)

    n = len(results)
    wins = sum(r["ours_won"] for r in results)
    p = wins / n
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    as_black = [r for r in results if r["ours_black"]]
    as_white = [r for r in results if not r["ours_black"]]
    summary = {
        "model": args.model, "opponent": f"GNU Go 3.8 level {args.level}", "games": n,
        "wins": wins, "win_rate": round(p, 3), "ci95": [round(max(0, p - 1.96 * se), 3), round(min(1, p + 1.96 * se), 3)],
        "as_black": f"{sum(r['ours_won'] for r in as_black)}/{len(as_black)}",
        "as_white": f"{sum(r['ours_won'] for r in as_white)}/{len(as_white)}",
        "avg_moves": round(sum(r["moves"] for r in results) / n, 1),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "games": results}, indent=1))


if __name__ == "__main__":
    main()
