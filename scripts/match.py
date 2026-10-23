"""Engine-vs-engine matches. GNU Go referees every game (final_score removes dead stones).

Player specs:
    policy:<ckpt>                       raw policy, greedy
    mcts:<ckpt>:<playouts>[:<value_ckpt>]   PUCT search; value from <value_ckpt> or the same net
    gnugo:<level>
    gtp:<command line>                  any external GTP engine, e.g.
                                        "gtp:katago gtp -model m.bin.gz -config c.cfg"

    python scripts/match.py mcts:checkpoints/sl_6x96/final.pt:400 policy:checkpoints/sl_6x96/final.pt --pairs 10

To avoid replaying one deterministic game, each pair starts from a short opening sampled from
the SL policy, and the two games of a pair swap colours.
"""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

import argparse
import hashlib
import queue
from collections import deque
import json
import math
import shlex
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np

from alphago import Board, Color
from alphago.gtp import PolicyPlayer, action_to_vertex, load_net, vertex_to_action
from alphago.mcts import MCTSPlayer
from match_gnugo import sgf_point

GPU_LOCK = threading.Lock()


@lru_cache(maxsize=None)
def net(path):
    return load_net(path)


class Engine:
    """Uniform wrapper: genmove(board, gnugo_referee) -> action."""

    def __init__(self, spec, komi, seed=0, timeout=90):
        self.spec = spec
        self.timeout = timeout
        spec = spec.replace("{seed}", str(seed))
        kind, *rest = spec.split(":")
        self.kind = kind
        if kind == "policy":
            self.player = PolicyPlayer(net(rest[0]))
        elif kind == "mcts":
            extra = rest[2:]
            value = net(extra.pop(0)) if extra and "=" not in extra[0] else None
            options = dict(part.split("=", 1) for part in extra)
            if set(options) - {"sym", "sec", "batch"}:
                raise ValueError(f"unknown MCTS options: {options}")
            self.player = MCTSPlayer(net(rest[0]), value, playouts=int(rest[1]),
                                     batch=int(options.get("batch", "16")),
                                     value_symmetries=int(options.get("sym", "1")),
                                     time_limit=float(options["sec"]) if "sec" in options else None)
        elif kind == "gnugo":
            self.level = int(rest[0])
        elif kind == "gtp":
            self.proc = subprocess.Popen(shlex.split(spec[len("gtp:"):]), stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            self.responses = queue.Queue()
            self.stderr_tail = deque(maxlen=20)
            def stdout_reader():
                for line in self.proc.stdout:
                    self.responses.put(line)
                self.responses.put(None)
            def stderr_reader():
                for line in self.proc.stderr:
                    self.stderr_tail.append(line.rstrip())
            threading.Thread(target=stdout_reader, daemon=True).start()
            threading.Thread(target=stderr_reader, daemon=True).start()
            self.sent = 0
            try:
                for c in ("boardsize 19", "clear_board", f"komi {komi}"):
                    self.cmd(c)
            except Exception:
                self.close()
                raise
        else:
            raise ValueError(spec)

    def cmd(self, c):
        try:
            self.proc.stdin.write(c + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"GTP process closed input: {self.spec}") from e
        lines = []
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                line = self.responses.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                self.proc.kill()
                raise TimeoutError(f"GTP timed out: {c}; stderr: {list(self.stderr_tail)}")
            if line is None:
                raise RuntimeError(f"{self.spec} exited: {list(self.stderr_tail)}")
            if line.strip() == "" and lines:
                break
            if line.strip():
                lines.append(line.strip())
        resp = " ".join(lines)
        if not resp.startswith("="):
            raise RuntimeError(f"{self.spec}: {c} -> {resp}")
        return resp[1:].strip()

    def close(self):
        if self.kind == "gtp":
            try:
                self.proc.stdin.write("quit\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
            for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                try:
                    stream.close()
                except (BrokenPipeError, OSError):
                    pass

    def genmove(self, board, gnugo_proc, moves):
        if self.kind == "gtp":
            for i in range(self.sent, len(moves)):
                self.cmd(f"play {'B' if i % 2 == 0 else 'W'} {action_to_vertex(moves[i], 19)}")
            v = self.cmd(f"genmove {'B' if board.to_move == Color.BLACK else 'W'}")
            self.sent = len(moves) + 1
            return "resign" if v.lower() == "resign" else vertex_to_action(v, 19)
        if self.kind == "gnugo":
            v = gnugo_proc.cmd(f"genmove {'B' if board.to_move == Color.BLACK else 'W'}")
            return "resign" if v.lower() == "resign" else vertex_to_action(v, 19)
        with GPU_LOCK:
            started = time.perf_counter()
            move = self.player.genmove(board)
            self.last_compute_seconds = time.perf_counter() - started
            return move


def sample_opening(moves, seed):
    rng = np.random.default_rng(seed)
    p = PolicyPlayer(net(OPENING_NET), temperature=1.0, seed=int(rng.integers(1 << 31)))
    b = Board(19)
    out = []
    for _ in range(moves):
        with GPU_LOCK:
            a = p.genmove(b)
        b.play(a)
        out.append(a)
    return out


def play_game(idx, a_spec, b_spec, a_black, opening, args, out_dir):
    level = next((int(s.split(":")[1]) for s in (a_spec, b_spec) if s.startswith("gnugo")), 10)
    g = Engine(f"gtp:gnugo --mode gtp --level {level} --chinese-rules --positional-superko",
               args.komi, timeout=args.gtp_timeout)
    eng_a = eng_b = None
    board = Board(19, args.komi, superko=True)
    moves, result = [], None
    timings = {"A": [], "B": []}
    visits = {"A": [], "B": []}
    try:
        eng_a = Engine(a_spec, args.komi, seed=args.seed + idx // 2, timeout=args.gtp_timeout)
        eng_b = Engine(b_spec, args.komi, seed=args.seed + idx // 2, timeout=args.gtp_timeout)
        g.cmd("boardsize 19")
        g.cmd("clear_board")
        for a in opening:
            g.cmd(f"play {'B' if board.to_move == Color.BLACK else 'W'} {action_to_vertex(a, 19)}")
            board.play(a)
            moves.append(a)
        while not board.game_over and board.move_number < args.max_moves:
            black = board.to_move == Color.BLACK
            eng = eng_a if black == a_black else eng_b
            side = "A" if eng is eng_a else "B"
            started = time.perf_counter()
            a = eng.genmove(board, g, moves)
            timings[side].append(getattr(eng, "last_compute_seconds", time.perf_counter() - started))
            root = getattr(getattr(eng, "player", None), "last_root", None)
            if root is not None:
                visits[side].append(float(root.N.sum()))
            if a == "resign":
                result = "W+R" if black else "B+R"
                break
            if eng.kind != "gnugo":
                g.cmd(f"play {'B' if black else 'W'} {action_to_vertex(a, 19)}")
            board.play(a)
            moves.append(a)
        capped = result is None and not board.game_over and board.move_number >= args.max_moves
        if result is None:
            # A forced move cap is an unfinished game, never an adjudicated win/loss.
            result = "Void" if capped else g.cmd("final_score")
    finally:
        g.close()
        if eng_a is not None:
            eng_a.close()
        if eng_b is not None:
            eng_b.close()
    draw = result.strip() in ("0", "Draw", "Jigo")
    black_won = result.upper().startswith("B+")
    a_won = None if capped else (not draw and black_won == a_black)
    body = "".join(f";{'B' if i % 2 == 0 else 'W'}[{sgf_point(m)}]" for i, m in enumerate(moves))
    pb, pw = (a_spec, b_spec) if a_black else (b_spec, a_spec)
    (out_dir / f"game{idx:03d}.sgf").write_text(
        f"(;GM[1]FF[4]SZ[19]KM[{args.komi}]RU[Chinese]PB[{pb}]PW[{pw}]RE[{result}]{body})")
    record = {"game": idx, "pair": idx // 2, "a_black": a_black, "result": result,
              "a_won": a_won, "a_score": None if capped else (0.5 if draw else float(a_won)), "moves": len(moves),
              "capped": capped, "move_seconds": timings, "search_visits": visits}
    atomic_json(out_dir / f"game{idx:03d}.json", record)
    return record


def wilson_interval(wins, games, z=1.96):
    """Binomial interval that remains meaningful for zero losses or zero wins."""
    p = wins / games
    den = 1 + z * z / games
    mid = (p + z * z / (2 * games)) / den
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / den
    return [round(max(0., mid - half), 3), round(min(1., mid + half), 3)]


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def make_summary(results, a, b, minutes):
    eligible = [r for r in results if not r["capped"]]
    n = len(eligible)
    w = sum(r["a_won"] for r in eligible)
    score = sum(r["a_score"] for r in eligible) / n if n else None
    pairs = {}
    for r in eligible:
        pairs.setdefault(r["pair"], []).append(r["a_score"])
    paired = np.array([np.mean(v) for v in pairs.values() if len(v) == 2])
    pair_ci = None
    if len(paired) >= 2:
        rng = np.random.default_rng(123)
        means = paired[rng.integers(len(paired), size=(10000, len(paired)))].mean(1)
        pair_ci = np.quantile(means, [0.025, 0.975]).round(3).tolist()
    summary = {"A": a, "B": b, "games": n, "completed_games": len(results), "capped_games": len(results)-n,
               "A_wins": w, "draws": sum(r["a_score"] == 0.5 for r in eligible),
               "A_win_rate": round(score, 4) if n else None,
               "ci95": wilson_interval(sum(r["a_score"] for r in eligible), n) if n else None,
               "paired_bootstrap_ci95": pair_ci, "complete_pairs": len(paired),
               "elo_A_minus_B": round(400 * math.log10(score / (1-score))) if score and score < 1 else None,
               "avg_moves": round(float(np.mean([r["moves"] for r in results])), 1) if results else None,
               "minutes": round(minutes, 2)}
    for side in ("A", "B"):
        times = [t for r in results for t in r["move_seconds"][side]]
        visits = [v for r in results for v in r["search_visits"][side]]
        summary[side + "_mean_move_seconds"] = round(float(np.mean(times)), 4) if times else None
        summary[side + "_p95_move_seconds"] = round(float(np.quantile(times, .95)), 4) if times else None
        summary[side + "_mean_visits"] = round(float(np.mean(visits)), 1) if visits else None
    return summary


def main():
    global OPENING_NET
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--opening-moves", type=int, default=6)
    ap.add_argument("--opening-net", default="checkpoints/sl_6x96/final.pt")
    ap.add_argument("--komi", type=float, default=7.5)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--max-moves", type=int, default=500)
    ap.add_argument("--gtp-timeout", type=float, default=90)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    if args.pairs < 1 or args.parallel < 1:
        ap.error("pairs and parallel must be positive")
    OPENING_NET = args.opening_net
    name = args.name or f"{args.a}_vs_{args.b}".replace("/", "-").replace(":", "_")[:120]
    out_dir = Path("runs") / "matches" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {k: v for k, v in vars(args).items() if k not in ("resume", "parallel", "name")}
    manifest["source_sha256"] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                 for p in ("alphago/mcts.py", "scripts/match.py")}
    mpath = out_dir / "manifest.json"
    if mpath.exists():
        if not args.resume:
            raise ValueError("run already exists; use --resume (old data will never be overwritten)")
        if json.loads(mpath.read_text()) != manifest:
            raise ValueError("run conditions or code changed; choose a new name")
    elif list(out_dir.glob("game*.sgf")):
        raise ValueError("legacy run without manifest; choose a new name")
    else:
        atomic_json(mpath, manifest)
    openings = [sample_opening(args.opening_moves, args.seed + j) for j in range(args.pairs)]
    results = [json.loads(p.read_text()) for p in sorted(out_dir.glob("game*.json"))]
    done = {r["game"] for r in results}
    previous = json.loads((out_dir / "summary.json").read_text())["summary"]["minutes"] if (out_dir / "summary.json").exists() else 0
    t0 = time.time()
    with ThreadPoolExecutor(args.parallel) as ex:
        futs = [ex.submit(play_game, 2*j+k, args.a, args.b, k == 0, openings[j], args, out_dir)
                for j in range(args.pairs) for k in range(2) if 2*j+k not in done]
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            results.sort(key=lambda x: x["game"])
            summary = make_summary(results, args.a, args.b, previous + (time.time()-t0)/60)
            atomic_json(out_dir / "summary.json", {"summary": summary, "games": results})
            print(json.dumps({"game": r["game"], "result": r["result"], "A_won": r["a_won"],
                              "done": len(results), "total": 2*args.pairs, "A_score": summary["A_win_rate"]}), flush=True)
    summary = make_summary(results, args.a, args.b, previous + (time.time()-t0)/60)
    atomic_json(out_dir / "summary.json", {"summary": summary, "games": results})
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
