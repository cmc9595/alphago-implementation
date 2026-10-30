"""Fast MCTS self-play: many games in parallel, one big network batch per round.

Same data format as selfplay_cap_pilot.py (game_*.npz/.json/.sgf), so train_search.py
reads it unchanged. GNU Go adjudication runs in background threads while search continues.

  --profile N   play N games and print where time goes (select / evaluate / backup / judge)
"""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.2')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.15')
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from alphago.gtp import load_net
from alphago.mcts import NetEvaluator
from alphago.parallel_selfplay import ParallelConfig, ParallelSelfPlay
from alphago.search_learning import SearchPlayConfig, save_game
from search_policy_iteration import Referee, save_sgf
from value_source_ab import write_json, memory_guard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True)
    ap.add_argument('--games', type=int, default=64)
    ap.add_argument('--model', default='checkpoints/shared_supervised_6x96_20261018/best.pt')
    ap.add_argument('--seed', type=int, default=20263019)
    ap.add_argument('--parallel-games', type=int, default=32)
    ap.add_argument('--leaves-per-game', type=int, default=8)
    ap.add_argument('--max-eval-batch', type=int, default=512)
    ap.add_argument('--reuse-tree', action='store_true')
    ap.add_argument('--resign', action='store_true')
    ap.add_argument('--judge-threads', type=int, default=3)
    ap.add_argument('--max-moves', type=int, default=600)
    ap.add_argument('--visits', type=int, default=400, help='full (training) search visits')
    ap.add_argument('--fast-visits', type=int, default=100)
    ap.add_argument('--full-prob', type=float, default=.25, help='fraction of moves searched with --visits (training moves)')
    args = ap.parse_args(); os.chdir(ROOT)
    out = ROOT / 'runs/experiments' / args.run; out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    device = torch.device('mps'); torch.mps.set_per_process_memory_fraction(.2)
    memory_guard()
    net = load_net(str(ROOT / args.model), device)
    evaluator = NetEvaluator(net, device=device)
    play = SearchPlayConfig(visits=args.visits, fast_visits=args.fast_visits, full_search_probability=args.full_prob, batch=32, max_moves=args.max_moves)
    cfg = ParallelConfig(play=play, parallel_games=args.parallel_games, leaves_per_game=args.leaves_per_game,
                         max_eval_batch=args.max_eval_batch, reuse_tree=args.reuse_tree, resign=args.resign)
    write_json(out / 'manifest.json', {'args': vars(args), 'config': asdict(cfg)})

    local = threading.local(); referees = []
    def judge(game, rec):
        if not hasattr(local, 'ref'):
            local.ref = Referee(play.komi); referees.append(local.ref)
        t0 = time.monotonic()
        try:
            score, dead, own = local.ref.judge(rec['moves'].tolist())
            rec.update(judged=True, score=float(score), dead=np.asarray(dead, np.int16),
                       ownership=np.asarray(own, np.int8).reshape(-1))
        except Exception as exc:
            rec['judge_error'] = repr(exc)
        return time.monotonic() - t0

    pool = ThreadPoolExecutor(args.judge_threads); futures = []; done = []; lock = threading.Lock()
    def finalize(g, rec):
        judge_s = 0.
        if rec['resigned']:
            # Winner known; no ownership (masked to 0). Kept for policy, value label from resignation.
            rec.update(judged=True, score=float(g.resigned_winner))
        elif not rec['capped']:
            judge_s = judge(g, rec)
        rec.update(seed=g.seed)
        valid = bool(rec['judged'] and not rec['capped'])
        path = out / f'game_{g.gid:04d}.npz'
        save_game(path, rec)
        save_sgf(path.with_suffix('.sgf'), rec['moves'], rec['score'] if valid else None, 'parallel self-play', '400 or 100')
        detail = {'game': g.gid, 'moves': len(rec['moves']), 'wall_seconds': time.monotonic() - g.started,
                  'judge_seconds': judge_s, 'full_search_moves': int(rec['train_mask'].sum()),
                  'training_positions': int(rec['train_mask'].sum()) if valid else 0,
                  'total_visits': int(g.total_visits), 'valid': valid, 'resigned': bool(rec['resigned']),
                  'resign_enabled': bool(rec['resign_enabled'])}
        write_json(path.with_suffix('.json'), detail)
        with lock: done.append(detail)

    def on_finish(g):
        futures.append(pool.submit(finalize, g, g.record()))
        if len(futures) % 8 == 0: memory_guard()

    seeds = [(i, args.seed + i) for i in range(args.games) if not (out / f'game_{i:04d}.json').exists()]
    evaluator([__import__('alphago').Board(19, 7.5, superko=True)] * 64); torch.mps.synchronize()
    timers = {}; t0 = time.monotonic()
    ParallelSelfPlay(evaluator, cfg, seeds, on_finish, timers).run()
    for f in futures: f.result()
    wall = time.monotonic() - t0
    pool.shutdown(); [r.close() for r in referees]
    moves = sum(d['moves'] for d in done)
    summary = {'games': len(done), 'wall_seconds': wall, 'seconds_per_game': wall / max(1, len(done)),
               'moves': moves, 'visits': sum(d['total_visits'] for d in done),
               'visits_per_second': sum(d['total_visits'] for d in done) / wall,
               'training_positions': sum(d['training_positions'] for d in done),
               'valid_games': sum(d['valid'] for d in done), 'resigned_games': sum(d['resigned'] for d in done),
               'time_split_seconds': {k: round(v, 1) for k, v in timers.items()},
               'judge_seconds_total_background': sum(d['judge_seconds'] for d in done)}
    write_json(out / 'summary.json', summary); print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
