"""Fast A/B match: many games at once, leaves batched per model, GNU Go judging in threads.

Same rules as match.py for mcts players (fresh tree each move, `visits` playouts, argmax N,
no noise), same SL-sampled openings with colour-swapped pairs, same output files
(game###.json / .sgf, summary.json), so progress_ab.py and the round pipelines read it.
Difference: each tree collects `leaves_per_game` leaves per round (default 8) instead of 32.

  python scripts/match_parallel.py A.pt B.pt --visits 400 --pairs 50 --name ab_x
"""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.3')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.25')
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import match as M
from alphago import Board, Color
from alphago.gtp import load_net
from alphago.mcts import NetEvaluator
from alphago.parallel_selfplay import make_nodes
from match_gnugo import sgf_point
from search_policy_iteration import Referee


class MatchGame:
    def __init__(self, idx, a_black, opening, cfg):
        self.idx, self.a_black, self.cfg = idx, a_black, cfg
        self.board = Board(19, cfg.komi, superko=True)
        self.moves = []
        for a in opening:
            self.board.play(a); self.moves.append(a)
        self.root = None
        self.visits = {'A': [], 'B': []}

    @property
    def side(self):
        return 'A' if (self.board.to_move == Color.BLACK) == self.a_black else 'B'

    @property
    def finished(self):
        return self.board.game_over or self.board.move_number >= self.cfg.max_moves

    def select_leaves(self, count, c_puct=1.5, vl=1.0, fpu=0.2):
        out = []
        for _ in range(count):
            node, b, path = self.root, self.board.copy(), []
            while node.moves.size:
                total = node.N.sum()
                parent_q = (node.W.sum() / total) if total > 0 else node.value
                q = np.where(node.N > 0, node.W / np.maximum(node.N, 1e-9), parent_q - fpu)
                i = int(np.argmax(q + c_puct * node.P * np.sqrt(total + 1) / (1 + node.N)))
                path.append((node, i)); node.N[i] += vl; node.W[i] -= vl
                b.play(int(node.moves[i]))
                child = node.children.get(i)
                if child is None:
                    out.append((path, b, None)); break
                node = child
            else:
                out.append((path, None, node))
        return out

    @staticmethod
    def backup(path, leaf, vl=1.0):
        parent, i = path[-1]
        parent.children.setdefault(i, leaf)
        val = -leaf.value
        for node, j in reversed(path):
            node.N[j] += 1 - vl; node.W[j] += val + vl
            val = -val

    def play_best(self):
        self.visits[self.side].append(float(self.root.N.sum()))
        a = int(self.root.moves[int(np.argmax(self.root.N))])
        self.board.play(a); self.moves.append(a); self.root = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--visits', type=int, default=400)
    ap.add_argument('--pairs', type=int, default=50)
    ap.add_argument('--parallel-games', type=int, default=32)
    ap.add_argument('--leaves-per-game', type=int, default=8)
    ap.add_argument('--opening-moves', type=int, default=6)
    ap.add_argument('--opening-net', default='checkpoints/sl_6x96/final.pt')
    ap.add_argument('--komi', type=float, default=7.5)
    ap.add_argument('--max-moves', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20261017)
    ap.add_argument('--judge-threads', type=int, default=3)
    ap.add_argument('--name', required=True)
    args = ap.parse_args(); os.chdir(ROOT)
    out = ROOT / 'runs/matches' / args.name; out.mkdir(parents=True, exist_ok=True)
    spec = lambda p: f'mcts:{p}:{args.visits}:parallel_leaves={args.leaves_per_game}'
    M.atomic_json(out / 'manifest.json', {**vars(args), 'A': spec(args.a), 'B': spec(args.b)})
    device = torch.device('mps'); torch.mps.set_per_process_memory_fraction(.25)
    def evaluator(spec_):  # "net.pt" (shared) or "policy.pt+value.pt" (separate value net)
        pol, _, val = spec_.partition('+')
        return NetEvaluator(load_net(str(ROOT / pol), device),
                            load_net(str(ROOT / val), device) if val else None, device=device)
    ev = {'A': evaluator(args.a), 'B': evaluator(args.b)}
    M.OPENING_NET = args.opening_net
    openings = [M.sample_opening(args.opening_moves, args.seed + j) for j in range(args.pairs)]
    done = {json.loads(p.read_text())['game'] for p in out.glob('game*.json')}
    queue_ = [(2 * j + k, k == 0, openings[j]) for j in range(args.pairs) for k in range(2) if 2 * j + k not in done]
    results = [json.loads(p.read_text()) for p in sorted(out.glob('game*.json'))]
    lock = threading.Lock(); local = threading.local(); refs = []; t0 = time.time()

    def finish(g):
        capped = not g.board.game_over
        if capped:
            result = 'Void'
        else:
            if not hasattr(local, 'ref'):
                local.ref = Referee(args.komi); refs.append(local.ref)
            score = local.ref.judge(g.moves)[0]
            result = '0' if score == 0 else f'{"B" if score > 0 else "W"}+{abs(score):g}'
        draw = result == '0'; black_won = result.startswith('B+')
        a_won = None if capped else (not draw and black_won == g.a_black)
        body = ''.join(f";{'B' if i % 2 == 0 else 'W'}[{sgf_point(m)}]" for i, m in enumerate(g.moves))
        pb, pw = (spec(args.a), spec(args.b)) if g.a_black else (spec(args.b), spec(args.a))
        (out / f'game{g.idx:03d}.sgf').write_text(f'(;GM[1]FF[4]SZ[19]KM[{args.komi}]RU[Chinese]PB[{pb}]PW[{pw}]RE[{result}]{body})')
        rec = {'game': g.idx, 'pair': g.idx // 2, 'a_black': g.a_black, 'result': result, 'a_won': a_won,
               'a_score': None if capped else (0.5 if draw else float(a_won)), 'moves': len(g.moves),
               'capped': capped, 'move_seconds': {'A': [], 'B': []}, 'search_visits': g.visits}
        M.atomic_json(out / f'game{g.idx:03d}.json', rec)
        with lock:
            results.append(rec); results.sort(key=lambda r: r['game'])
            s = M.make_summary(results, spec(args.a), spec(args.b), (time.time() - t0) / 60)
            M.atomic_json(out / 'summary.json', {'summary': s, 'games': results})
            print(json.dumps({'game': g.idx, 'result': result, 'A_won': a_won, 'done': len(results),
                              'total': 2 * args.pairs, 'A_score': s['A_win_rate']}), flush=True)

    pool = ThreadPoolExecutor(args.judge_threads); futures = []; games = []
    while queue_ or games:
        while queue_ and len(games) < args.parallel_games:
            games.append(MatchGame(*queue_.pop(0), args))
        req = {'A': [], 'B': []}
        for g in games:
            if g.root is None:
                req[g.side].append((g, None, g.board))
            else:
                for path, b, cached in g.select_leaves(min(args.leaves_per_game, args.visits - int(g.root.N.sum()))):
                    req[g.side].append((g, (path, cached), b))
        for side, items in req.items():
            boards = [b for _, _, b in items if b is not None]
            nodes = iter(make_nodes(boards, *ev[side](boards)) if boards else [])
            for g, leaf, b in items:
                node = next(nodes) if b is not None else leaf[1]
                if leaf is None:
                    g.root = node
                else:
                    g.backup(leaf[0], node)
        still = []
        for g in games:
            if g.root is not None and int(g.root.N.sum()) >= args.visits:
                g.play_best()
            if g.finished:
                futures.append(pool.submit(finish, g))
            else:
                still.append(g)
        games = still
    for f in futures: f.result()
    pool.shutdown(); [r.close() for r in refs]
    s = M.make_summary(results, spec(args.a), spec(args.b), (time.time() - t0) / 60)
    M.atomic_json(out / 'summary.json', {'summary': s, 'games': results}); print(json.dumps(s, indent=1))


if __name__ == '__main__':
    main()
