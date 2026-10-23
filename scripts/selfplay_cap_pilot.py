"""Generate only: randomized 400/100 search caps, resumable, no training or promotion."""
import argparse
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = 'selfplay_cap_400_100_20261019'
KST = ZoneInfo('Asia/Seoul')


def write_json(path, obj):
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp():
    return datetime.now(KST).isoformat(timespec='seconds')


def duration(seconds):
    m, s = divmod(max(0, int(seconds)), 60)
    h, m = divmod(m, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def progress(out, watch):
    while True:
        if watch: print('\033[2J\033[H', end='')
        if (out/'state.json').exists():
            s = json.loads((out/'state.json').read_text())
            done, total = s.get('games_done', 0), s['games']
            filled = int(30*done/total)
            print(f"400/100회 자가대국 시간 측정 · {s['status']}")
            print(f"[{'█'*filled}{'░'*(30-filled)}] {done}/{total}판 ({100*done/total:.1f}%)")
            print(f"현재 {s.get('game_id', 0)}판 · {s.get('move', 0)}수 · 학습 가능 국면 {s.get('training_positions', 0):,}개")
            print(f"유효 판정 {s.get('valid_games', 0)}판 · 제외 {s.get('invalid_games', 0)}판")
            print(f"누적 생성·판정 시간 {duration(s.get('total_seconds', 0))}")
            if done and s['status'] == 'selfplay':
                avg = s['total_seconds']/done
                remaining = avg*(total-done)
                print(f"완료 판 평균 {avg:.1f}초/판 · 남은 약 {duration(remaining)}")
                print('예상 종료', (datetime.now(KST)+timedelta(seconds=remaining)).strftime('%m-%d %H:%M:%S KST'))
                print('현재 진행 중인 판은 ETA 계산에서 미차감 · 절전/다른 GPU 작업 영향 포함 가능')
            if s.get('error'): print('오류:', s['error'])
            print('갱신', s['updated'])
            if (out/'summary.json').exists():
                result = json.loads((out/'summary.json').read_text())
                print(f"최종 평균 {result['mean_game_seconds']:.2f}초/판 · 전체 {duration(result['total_seconds'])}")
        else: print('아직 시작하지 않았습니다.')
        if not watch: break
        time.sleep(2)


def run(args, out):
    # Import the GPU runtime only for generation, never for a progress display.
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.12')
    os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.10')
    import numpy as np
    import torch
    from alphago.gtp import load_net
    from alphago.mcts import NetEvaluator
    from alphago.search_learning import SearchPlayConfig, generate_game, save_game
    from search_policy_iteration import Referee, save_sgf
    from value_source_ab import memory_guard
    from dataclasses import asdict

    out.mkdir(parents=True, exist_ok=True)
    locks = []
    # Acquire the existing experiment locks atomically; never overlap a training job.
    paths = [out/'runner.lock', ROOT/'runs/experiments/value_source_ab_20261018/experiment.lock',
             ROOT/'runs/experiments/search_iteration_400_20261018/runner.lock',
             ROOT/'runs/experiments/shared_supervised_6x96_20261018/pipeline.lock']
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, 'a'); locks.append(f)
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = SearchPlayConfig(visits=400, fast_visits=100, full_search_probability=.25,
                           batch=32, max_moves=args.max_moves)
    model_path = (ROOT/args.model).resolve()
    manifest = {'games': args.games, 'seed': args.seed, 'config': asdict(cfg),
                'producer_model': str(model_path), 'producer_sha256': digest(model_path),
                'source_sha256': {p: digest(ROOT/p) for p in ('alphago/search_learning.py', 'alphago/mcts.py',
                                      'scripts/selfplay_cap_pilot.py', 'scripts/search_policy_iteration.py',
                                      'alphago/gnugo.py', 'scripts/match.py')},
                'training': False, 'automatic_promotion': False,
                'fast_search': 'greedy, no root Dirichlet noise; excluded by train_mask',
                'referee': 'GNU Go level10 Chinese positional-superko, 30s command timeout',
                'timing': 'monotonic full game including feature extraction and GNU Go; model load excluded'}
    mp = out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Resume manifest differs; use a new run name')
    write_json(mp, manifest)
    (out/'runner.pid').write_text(str(os.getpid()))
    state = {'status': 'preparing', 'pid': os.getpid(), 'games': args.games, 'games_done': 0,
             'training_positions': 0, 'valid_games': 0, 'invalid_games': 0, 'total_seconds': 0.}
    def status(**kw):
        state.update(kw, updated=stamp()); write_json(out/'state.json', state)
    records = []
    def aggregate():
        valid = [r for r in records if r['valid']]
        status(games_done=len(records), valid_games=len(valid), invalid_games=len(records)-len(valid),
               training_positions=sum(r['training_positions'] for r in valid),
               total_seconds=sum(r['seconds'] for r in records))
    caffeine = None
    try:
        status()
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS unavailable; refusing CPU generation')
        torch.set_num_threads(4); torch.set_num_interop_threads(2)
        torch.mps.set_per_process_memory_fraction(.12)
        memory_guard()
        caffeine = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())])
        model = load_net(str(model_path), torch.device('mps'))
        evaluator = NetEvaluator(model, device=torch.device('mps'))
        # Unmeasured inference warmup; explicitly synchronize before timing.
        from alphago import Board
        evaluator([Board(19, 7.5, superko=True)]*32)
        torch.mps.synchronize()
        for game in range(args.games):
            gp, jp = out/f'game_{game:03d}.npz', out/f'game_{game:03d}.json'
            if gp.exists() and jp.exists():
                records.append(json.loads(jp.read_text())); aggregate(); continue
            status(status='selfplay', game_id=game+1, move=0)
            started = time.monotonic()
            visits_done = 0
            def on_move(ply, count):
                nonlocal visits_done
                visits_done += count
                if ply % 10 == 0:
                    memory_guard()
                    status(move=ply, current_game_visits=visits_done,
                           current_game_seconds=time.monotonic()-started)
            referee = Referee(cfg.komi)
            judge_seconds = 0.
            def judge(moves):
                nonlocal judge_seconds
                status(move=len(moves), phase='GNU Go判定')
                ts = time.monotonic()
                try: return referee.judge(moves)
                finally: judge_seconds = time.monotonic()-ts
            try:
                rec = generate_game(evaluator, cfg, np.random.default_rng(args.seed+game), judge, on_move)
            finally: referee.close()
            torch.mps.synchronize()
            elapsed = time.monotonic()-started
            rec.update(producer_model_sha256=manifest['producer_sha256'], seed=args.seed+game)
            valid = bool(rec['judged'] and not rec['capped'])
            detail = {'game': game, 'moves': len(rec['moves']), 'seconds': elapsed,
                      'search_and_setup_seconds': elapsed-judge_seconds, 'judge_seconds': judge_seconds,
                      'full_search_moves': int(rec['train_mask'].sum()),
                      'fast_search_moves': int((~rec['train_mask']).sum()),
                      'training_positions': int(rec['train_mask'].sum()) if valid else 0,
                      'total_visits': int(rec['visits'].sum()), 'valid': valid,
                      'capped': bool(rec['capped']), 'judged': bool(rec['judged']),
                      'judge_error': rec.get('judge_error'), 'finished_at': stamp()}
            save_game(gp, rec)
            save_sgf(gp.with_suffix('.sgf'), rec['moves'], rec['score'] if valid else None,
                     'Frozen shared model; randomized 25% 400 / 75% 100; train_mask full only', '400 or 100')
            write_json(jp, detail); records.append(detail)
            aggregate(); status(move=0, phase='selfplay')
            print(json.dumps(detail, ensure_ascii=False), flush=True)
        total = sum(r['seconds'] for r in records)
        total_moves = sum(r['moves'] for r in records)
        summary = {**state, 'status': 'completed', 'mean_game_seconds': total/len(records),
                   'median_game_seconds': float(np.median([r['seconds'] for r in records])),
                   'total_moves': total_moves, 'mean_move_seconds': total/total_moves,
                   'full_search_moves': sum(r['full_search_moves'] for r in records),
                   'fast_search_moves': sum(r['fast_search_moves'] for r in records),
                   'total_visits': sum(r['total_visits'] for r in records),
                   'total_judge_seconds': sum(r['judge_seconds'] for r in records),
                   'training': False, 'automatic_promotion': False,
                   'note': '32-game throughput pilot; invalid games excluded; other GPU work and sleep can affect time'}
        write_json(out/'summary.json', summary)
        status(status='completed', move=0)
    except BaseException as exc:
        status(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
        raise
    finally:
        if caffeine and caffeine.poll() is None: caffeine.terminate()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=DEFAULT_RUN)
    ap.add_argument('--games', type=int, default=32)
    ap.add_argument('--model', default='checkpoints/shared_supervised_6x96_20261018/best.pt')
    ap.add_argument('--seed', type=int, default=20261019)
    ap.add_argument('--max-moves', type=int, default=600)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--progress', action='store_true')
    args = ap.parse_args()
    if args.games < 1 or args.max_moves < 2: ap.error('Positive games and at least 2 max moves required')
    out = ROOT/'runs/experiments'/args.run
    if args.watch or args.progress:
        try: progress(out, args.watch)
        except KeyboardInterrupt: pass
    else: run(args, out)


if __name__ == '__main__': main()
