"""Bounded 400-visit search self-play -> policy/value updates -> fixed-visit gate.

Runs after the existing value-data experiment releases its lock. Never promotes
models to the live browser automatically. --watch displays progress and ETA.
"""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.10')
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from alphago import Board, Color
from alphago.gnugo import action as parse_action, ownership, parse_score
from alphago.gtp import PolicyPlayer, action_to_vertex, load_net
from alphago.mcts import MCTSPlayer, NetEvaluator
from alphago.search_learning import SearchData, SearchPlayConfig, generate_game, losses, save_game
from match import Engine, wilson_interval
from value_source_ab import memory_guard, write_json, digest, duration

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = 'search_iteration_400_20261018'

def save_sgf(path, moves, score=None, label='MCTS selfplay', visits=400):
    result = 'Void' if score is None else ('0' if score == 0 else f'{"B" if score > 0 else "W"}+{abs(score):g}')
    body = []
    for i, action in enumerate(moves):
        point = '' if action == 361 else chr(97 + action % 19) + chr(97 + 18 - action // 19)
        body.append(f';{"B" if i % 2 == 0 else "W"}[{point}]')
    Path(path).write_text(f'(;GM[1]FF[4]SZ[19]KM[7.5]RU[Chinese]RE[{result}]C[{label}; {visits} visits per searched move]' + ''.join(body) + ')')

def pid_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


class Referee:
    """Timeout-protected GNU Go, matching positional-superko/Chinese rules."""
    def __init__(self, komi):
        self.komi = komi
        self.engine = Engine('gtp:gnugo --mode gtp --level 10 --chinese-rules --positional-superko', komi, timeout=30)

    def judge(self, moves):
        self.engine.cmd('clear_board')
        for i, a in enumerate(moves):
            self.engine.cmd(f'play {"B" if i % 2 == 0 else "W"} {action_to_vertex(a, 19)}')
        score = parse_score(self.engine.cmd('final_score'))
        dead = [parse_action(v) for v in self.engine.cmd('final_status_list dead').split()]
        return score, dead, ownership(moves, dead, 19, self.komi)

    def close(self):
        self.engine.close()


def watch(run, loop):
    out = ROOT / 'runs/experiments' / run
    while True:
        if loop: print('\033[2J\033[H', end='')
        path = out / 'state.json'
        if path.exists():
            s = json.loads(path.read_text())
            print('MCTS 정책·가치 공동 학습 ·', s['status'])
            print('탐색:', s.get('visits', 400), '회/수 · 반복', s.get('iteration', 0), '/', s.get('iterations', 1))
            for key, label in [('games_done', '자가대국'), ('train_step', '공동 학습'), ('gate_done', '기존 엔진 상대 평가')]:
                total = s.get({'games_done': 'games', 'train_step': 'train_steps', 'gate_done': 'gate_games'}[key], 0)
                done = s.get(key, 0)
                if total:
                    filled = int(30 * min(1, done / total))
                    print(f'{label}: [{"█" * filled}{"░" * (30-filled)}] {done}/{total}')
            if s.get('status') == 'waiting_for_value_experiment':
                print('기존 가치망 데이터 비교 실험 종료를 기다립니다. GPU 중복 사용 없음.')
            if s.get('move'):
                print('현재 판:', s.get('game_id', 0), '·', s['move'], '수')
            if s.get('recent_game_seconds') and s.get('status') == 'selfplay':
                remaining = max(0, s['games'] - s.get('games_done', 0)) * s['recent_game_seconds']
                print('자가대국 단계 예상 남은 시간', duration(remaining), '· 종료', (datetime.now()+timedelta(seconds=remaining)).strftime('%m-%d %H:%M'))
                print('후속 학습·평가 시간은 별도입니다.')
            if s.get('metrics'): print('최근 학습:', json.dumps(s['metrics'], ensure_ascii=False))
            if s.get('error'): print('오류:', s['error'])
            print('갱신:', s.get('updated'))
        else:
            print('아직 시작하지 않았습니다.')
        if not loop: break
        time.sleep(10)


@torch.no_grad()
def evaluate_pair(policy, value, anchor, data, device, batch):
    policy.eval(); value.eval(); anchor.eval()
    totals, n = {}, 0
    for i in range(0, len(data.rows), batch):
        rows = data.rows[i:i+batch]
        _, metrics = losses(policy, value, anchor, data.batch(rows), device)
        for k, v in metrics.items(): totals[k] = totals.get(k, 0.) + float(v) * len(rows)
        n += len(rows)
    return {k: v / n for k, v in totals.items()}


def train_pair(policy_path, value_path, train_paths, val_paths, out, args, device, status):
    # Files are split by whole game before extracting any position.
    train = SearchData(train_paths)
    val = SearchData(val_paths)
    steps = max(1, math.ceil(len(train.rows) * args.epochs / args.train_batch))
    if steps > args.max_updates:
        steps = args.max_updates
    status(status='training', train_step=0, train_steps=steps,
           train_positions=len(train.rows), validation_positions=len(val.rows))
    policy = load_net(str(policy_path), device)
    value = load_net(str(value_path), device)
    anchor = load_net(str(ROOT / 'checkpoints/sl_6x96/final.pt'), device)
    for p in anchor.parameters(): p.requires_grad_(False)
    # Its old value head is unused by the separate-net engine and is not trained.
    for name, p in policy.named_parameters():
        if name.startswith('value_') or name.startswith('ownership_'): p.requires_grad_(False)
    groups = []
    for net, lr in [(policy, args.policy_lr), (value, args.value_lr)]:
        for decay in (False, True):
            params = [p for p in net.parameters() if p.requires_grad and (p.ndim > 1) == decay]
            groups.append({'params': params, 'lr': lr, 'initial_lr': lr, 'weight_decay': 1e-4 if decay else 0.})
    optimizer = torch.optim.SGD(groups, momentum=.9, nesterov=True)
    before = evaluate_pair(policy, value, anchor, val, device, args.train_batch)
    write_json(out / 'validation_before.json', before)
    rng = np.random.default_rng(args.seed + int(out.name.split('_')[-1]))
    torch.manual_seed(args.seed)
    order, cursor = rng.permutation(len(train.rows)), 0
    running = {}
    started = time.monotonic()
    for step in range(1, steps + 1):
        if cursor >= len(order): order, cursor = rng.permutation(len(train.rows)), 0
        selected = order[cursor:cursor + args.train_batch]
        cursor += len(selected)
        # BatchNorm requires at least two examples; a one-row tail joins next epoch.
        if len(selected) < 2:
            order, cursor = rng.permutation(len(train.rows)), args.train_batch
            selected = order[:args.train_batch]
        policy.train(); value.train(); anchor.eval()
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr'] * (.1 + .9 * .5 * (1 + math.cos(math.pi * (step - 1) / max(1, steps))))
        batch = train.batch(train.rows[selected], rng, augment=True)
        loss, metrics = losses(policy, value, anchor, batch, device, args.kl_weight, 1.5)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite joint loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], 5., error_if_nonfinite=True)
        optimizer.step()
        running = {k: float(v.detach()) for k, v in metrics.items()}
        if step % 10 == 0 or step == steps:
            rec = {'step': step, **running, 'grad_norm_before_clip': float(norm)}
            with open(out / 'train_metrics.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
            status(train_step=step, metrics=rec)
            memory_guard()
    after = evaluate_pair(policy, value, anchor, val, device, args.train_batch)
    write_json(out / 'validation_after.json', after)
    for label, net in [('policy', policy), ('value', value)]:
        path = out / f'{label}.pt'
        tmp = path.with_suffix('.tmp')
        torch.save({'model': net.state_dict(), 'config': vars(net.cfg), 'step': steps,
                    'source': 'MCTS visit policy + terminal outcome value', 'args': vars(args),
                    'validation': after, 'requires_separate_value': label == 'policy'}, tmp)
        tmp.replace(path)
    write_json(out / 'training_summary.json', {'steps': steps, 'positions': len(train.rows), 'seconds': time.monotonic()-started,
               'before': before, 'after': after, 'policy_lr': args.policy_lr, 'value_lr': args.value_lr})
    del policy, value, anchor, optimizer
    if device.type == 'mps': torch.mps.empty_cache()
    return out / 'policy.pt', out / 'value.pt'


def evaluate_gate(candidate_policy, candidate_value, out, args, device, status):
    baseline_p = load_net(str(ROOT/'checkpoints/sl_6x96/final.pt'), device)
    baseline_v = load_net(str(ROOT/'checkpoints/value_own/best.pt'), device)
    candidate_p, candidate_v = load_net(str(candidate_policy), device), load_net(str(candidate_value), device)
    players = [MCTSPlayer(candidate_p, candidate_v, playouts=args.visits, batch=args.search_batch),
               MCTSPlayer(baseline_p, baseline_v, playouts=args.visits, batch=args.search_batch)]
    records = []
    status(status='gate', gate_done=0, gate_games=args.gate_pairs*2, move=0)
    for pair in range(args.gate_pairs):
        opening = []
        board = Board(19, 7.5, superko=True)
        sampler = PolicyPlayer(baseline_p, temperature=1., seed=args.seed+pair)
        for _ in range(8):
            a = sampler.genmove(board); board.play(a); opening.append(a)
        for candidate_black in (True, False):
            idx = pair*2 + int(not candidate_black)
            path = out/f'gate_{idx:03d}.json'
            if path.exists():
                records.append(json.loads(path.read_text())); status(gate_done=len(records)); continue
            board = Board(19, 7.5, superko=True)
            moves = opening.copy()
            for a in moves: board.play(a)
            seconds, counts = [0., 0.], [0, 0]
            while not board.game_over and len(moves) < args.max_moves:
                side = 0 if (board.to_move == Color.BLACK) == candidate_black else 1
                started = time.monotonic()
                a = players[side].genmove(board)
                seconds[side] += time.monotonic()-started
                counts[side] += int(players[side].last_root.N.sum())
                board.play(a); moves.append(a)
                if len(moves) % 10 == 0: status(move=len(moves), game_id=idx+1)
            capped = not board.game_over
            score, error = None, None
            if not capped:
                ref = Referee(7.5)
                try:
                    score, _, _ = ref.judge(moves)
                except Exception as exc: error = repr(exc)
                finally: ref.close()
            result = None if score is None else (.5 if score == 0 else float((score > 0) == candidate_black))
            rec = {'game': idx, 'pair': pair, 'candidate_black': candidate_black, 'opening': opening, 'moves': moves,
                   'capped': capped, 'judge_error': error, 'score_black': score, 'candidate_score': result,
                   'search_visits': counts, 'compute_seconds': seconds, 'visits_per_move': args.visits}
            write_json(path, rec); records.append(rec); status(gate_done=len(records), move=0)
            save_sgf(path.with_suffix('.sgf'), moves, score, 'Candidate vs frozen baseline; first 8 moves shared opening', args.visits)
            memory_guard()
    scored = [r['candidate_score'] for r in records if r['candidate_score'] is not None]
    summary = {'games': len(records), 'scored_games': len(scored), 'void_games': len(records)-len(scored),
               'candidate_score': float(np.mean(scored)) if scored else None,
               'wilson_95': wilson_interval(sum(scored), len(scored)) if scored else None,
               'automatic_promotion': False, 'note': 'Small pilot: no definitive strength claim; both sides use identical visits/settings'}
    write_json(out/'gate_summary.json', summary)
    return summary


def run(args):
    out = ROOT / 'runs/experiments' / args.run
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out/'runner.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'preparing', 'pid': os.getpid(), 'games': args.games, 'iterations': args.iterations,
             'iteration': 0, 'visits': args.visits}
    def status(**changes):
        state.update(changes, updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(out/'state.json', state)
    status()
    try:
        wait_lock_path = ROOT/'runs/experiments/value_source_ab_20261018/experiment.lock'
        wait_lock_path.parent.mkdir(parents=True, exist_ok=True)
        wait_lock = open(wait_lock_path, 'a')
        while True:
            try:
                fcntl.flock(wait_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                status(status='waiting_for_value_experiment')
                time.sleep(30)
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        if device.type != 'mps': raise RuntimeError('MPS unavailable; refusing accidental CPU pilot')
        torch.set_num_threads(4); torch.set_num_interop_threads(2)
        torch.mps.set_per_process_memory_fraction(.12)
        memory_guard()
        policy_path = ROOT/'checkpoints/sl_6x96/final.pt'
        value_path = ROOT/'checkpoints/value_own/best.pt'
        config = SearchPlayConfig(visits=args.visits, batch=args.search_batch, max_moves=args.max_moves)
        manifest = {'args': vars(args), 'search': asdict(config), 'initial_policy_sha256': digest(policy_path),
                    'initial_value_sha256': digest(value_path), 'policy_target': 'normalized root visits, including pass',
                    'value_target': 'GNU Go terminal outcome, side-to-move', 'ownership_weight': 1.5,
                    'split': 'game index modulo 4 == 3 held out permanently',
                    'rules': 'Chinese komi7.5 positional-superko, no resignation, capped/failed games excluded',
                    'automatic_promotion': False}
        path = out/'manifest.json'
        if path.exists() and json.loads(path.read_text()) != manifest: raise ValueError('Resume configuration changed')
        write_json(path, manifest)
        for iteration in range(1, args.iterations+1):
            cycle = out/f'iteration_{iteration:03d}'; cycle.mkdir(exist_ok=True)
            complete = cycle/'complete.json'
            if complete.exists():
                policy_path, value_path = cycle/'policy.pt', cycle/'value.pt'
                continue
            status(status='selfplay', iteration=iteration, games_done=0, train_step=0, train_steps=0,
                   gate_done=0, gate_games=args.gate_pairs*2, move=0)
            policy, value = load_net(str(policy_path), device), load_net(str(value_path), device)
            evaluator = NetEvaluator(policy, value, device)
            durations = []
            for game in range(args.games):
                game_path = cycle/f'selfplay_{game:03d}.npz'
                if game_path.exists():
                    status(games_done=game+1); continue
                status(game_id=game+1, move=0)
                started = time.monotonic()
                def on_move(ply, count):
                    if ply % 10 == 0: status(move=ply, visits_completed_in_game=ply*count)
                referee = Referee(7.5)
                try:
                    record = generate_game(evaluator, config, np.random.default_rng(args.seed+iteration*100000+game), referee.judge, on_move)
                finally: referee.close()
                record['producer_policy_sha256'] = digest(policy_path)
                record['producer_value_sha256'] = digest(value_path)
                record['seed'] = args.seed+iteration*100000+game
                record['visits_per_move'] = args.visits
                save_game(game_path, record)
                save_sgf(game_path.with_suffix('.sgf'), record['moves'], record['score'] if record['judged'] else None,
                         'Frozen producer selfplay; root Dirichlet noise; no resignation', args.visits)
                durations.append(time.monotonic()-started)
                status(games_done=game+1, recent_game_seconds=float(np.mean(durations[-5:])), move=0,
                       last_game_capped=record['capped'], last_game_judged=record['judged'])
                print(json.dumps({'iteration': iteration, 'game': game, 'moves': len(record['moves']),
                      'capped': record['capped'], 'judged': record['judged'], 'seconds': durations[-1]}), flush=True)
                memory_guard()
            del evaluator, policy, value
            torch.mps.empty_cache()
            train_paths, val_paths = [], []
            # Replay previous cycles without leaking any previously held-out game.
            for old in range(1, iteration+1):
                for path in sorted((out/f'iteration_{old:03d}').glob('selfplay_*.npz')):
                    idx = int(path.stem.split('_')[-1])
                    (val_paths if idx % 4 == 3 else train_paths).append(path)
            write_json(cycle/'data_split.json', {'train': [str(p.relative_to(ROOT)) for p in train_paths],
                                               'validation': [str(p.relative_to(ROOT)) for p in val_paths]})
            # Training restarts from this cycle's frozen producer on interruption.
            # Completed selfplay games are reused, never overwritten.
            if not (cycle/'training_summary.json').exists():
                policy_path, value_path = train_pair(policy_path, value_path, train_paths, val_paths, cycle, args, device, status)
            else:
                policy_path, value_path = cycle/'policy.pt', cycle/'value.pt'
            gate = evaluate_gate(policy_path, value_path, cycle, args, device, status)
            write_json(complete, {'gate': gate, 'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')})
        status(status='completed', move=0)
    except BaseException as exc:
        status(status='failed', error=repr(exc))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=DEFAULT_RUN)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--progress', action='store_true')
    ap.add_argument('--visits', type=int, default=400)
    ap.add_argument('--search-batch', type=int, default=32)
    ap.add_argument('--games', type=int, default=16)
    ap.add_argument('--iterations', type=int, default=1)
    ap.add_argument('--max-moves', type=int, default=600)
    ap.add_argument('--epochs', type=float, default=2.)
    ap.add_argument('--max-updates', type=int, default=500)
    ap.add_argument('--train-batch', type=int, default=32)
    ap.add_argument('--policy-lr', type=float, default=.0005)
    ap.add_argument('--value-lr', type=float, default=.001)
    ap.add_argument('--kl-weight', type=float, default=.05)
    ap.add_argument('--gate-pairs', type=int, default=5)
    ap.add_argument('--seed', type=int, default=20261018)
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.watch or args.progress:
        watch(args.run, args.watch)
    else:
        if args.games < 8 or args.games % 4: ap.error('Use at least 8 games, divisible by 4, for game-held-out validation')
        if args.visits < 400: ap.error('Pilot requires at least 400 visits')
        if args.epochs <= 0 or args.train_batch < 2 or args.iterations < 1 or args.gate_pairs < 1: ap.error('Invalid training budget')
        run(args)


if __name__ == '__main__': main()
