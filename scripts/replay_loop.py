"""Three-cycle replay pilot. Single GPU job at a time; resume by rerunning.

No browser replacement, destructive data eviction, or automatic Git changes.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

from alphago.replay import atomic_json, build_pool, sha256

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = 'replay_pilot_1200v_p06_20261120'
BASE = 'checkpoints/depth_gpool_6x96_20261117/final.pt'
FIXED = 'runs/datasets/round6_fixed1000/manifest.json'
SOURCES = [
    ('selfplay_round3_800v_p0625_20000_seed20297031', False),
    ('selfplay_round4_800v_p0625_20000_seed20407102', False),
    ('selfplay_round4_800v_p0625_20000_seed20437102', False),
    ('selfplay_round5_1600v_p125_f200_seed20507105', True),
    ('selfplay_round6_1600v_p125_f200_seed20630000', True),
    ('selfplay_round7_gpool6_800v_p03_f200_30000_seed20730000', True),
]


def cycle_search(run, cycle, schedule):
    """Prospective changes create new batches; frozen/completed contexts stay intact."""
    override = schedule.get(str(cycle), {})
    if set(override) - {'visits', 'seed'}: raise ValueError('Unknown search override')
    visits = override.get('visits', 1200)
    seed = override.get('seed', 20800000 + cycle * 100000)
    if not isinstance(visits, int) or visits < 200 or not isinstance(seed, int) or seed < 0:
        raise ValueError('Invalid search override')
    name = f'{run}_selfplay_c{cycle:02d}'
    if visits != 1200: name += f'_v{visits}'
    return name, visits, seed


def match_statistics(directories, seed=20261120):
    scores, pair_scores = [], []
    for directory in directories:
        pairs = {}
        for p in sorted(Path(directory).glob('game*.json')):
            r = json.loads(p.read_text())
            if r.get('capped') or r.get('a_score') is None: continue
            scores.append(float(r['a_score']))
            pairs.setdefault(r['pair'], []).append(float(r['a_score']))
        pair_scores.extend(float(np.mean(s)) for s in pairs.values() if len(s) == 2)
    n = len(scores)
    rate = float(np.mean(scores)) if n else 0.
    z = 1.96; den = 1 + z * z / max(1, n)
    center = (rate + z * z / (2 * max(1, n))) / den
    half = z * math.sqrt(rate * (1 - rate) / max(1, n) + z * z / (4 * max(1, n)**2)) / den
    interval = [0., 1.]
    if pair_scores:
        a = np.asarray(pair_scores)
        rng = np.random.default_rng(seed)
        samples = a[rng.integers(0, len(a), (10000, len(a)))].mean(1)
        interval = np.quantile(samples, [.025, .975]).tolist()
    return {'games': n, 'wins': sum(s == 1 for s in scores), 'draws': sum(s == .5 for s in scores),
            'win_rate': rate, 'wilson95': [max(0., center - half), min(1., center + half)],
            'complete_pairs': len(pair_scores), 'paired_bootstrap95': interval}


def promotion_decision(screen, confirmation):
    screened = screen['games'] >= 95 and screen['win_rate'] >= .5
    # Confirmation is an independent, fixed 200-game batch, not repeated peeking.
    return bool(screened and confirmation and confirmation['games'] >= 190
                and confirmation['complete_pairs'] >= 95 and confirmation['wilson95'][0] > .5
                and confirmation['paired_bootstrap95'][0] > .5)


class Pilot:
    def __init__(self, args):
        self.args = args
        self.out = ROOT / 'runs/replay' / args.run
        self.out.mkdir(parents=True, exist_ok=True)
        self.state = {'run': args.run, 'cycle': 0, 'phase': 'preparing', 'cycles': 3}
        self.state_path = self.out / 'state.json'

    def update(self, **values):
        self.state.update(values, updated_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        atomic_json(self.state_path, self.state)

    def child(self, command, stage, pidfile=None):
        expected = {'command': command}
        marker = stage.with_suffix('.command.json')
        if marker.exists() and json.loads(marker.read_text()) != expected:
            raise ValueError(f'Changed child command: {marker}')
        atomic_json(marker, expected)
        print('RUN', ' '.join(command), flush=True)
        with stage.with_suffix('.log').open('ab') as log:
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if pidfile: Path(pidfile).write_text(str(p.pid) + '\n')
            self.update(child_pid=p.pid, child_log=str(stage.with_suffix('.log').relative_to(ROOT)))
            try:
                while True:
                    try:
                        code = p.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        if shutil.disk_usage(ROOT).free < 10 * 1024**3:
                            raise RuntimeError('Disk free below 10 GiB')
                if code: raise RuntimeError(f'Child exited {code}; see {stage.with_suffix(".log")}')
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    try: p.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL); p.wait()
            self.update(child_pid=None)

    def run_match(self, cycle, candidate, champion, phase, pairs, seed):
        name = f'{self.args.run}_c{cycle:02d}_{phase}'
        directory = ROOT / 'runs/matches' / name
        self.update(phase=phase, match=name, match_total=pairs * 2)
        summary = directory / 'summary.json'
        complete = summary.exists() and json.loads(summary.read_text())['summary']['completed_games'] == pairs * 2
        if not complete:
            self.child([sys.executable, '-u', 'scripts/match_parallel.py', str(candidate), str(champion),
                        '--opening-net', str(ROOT / BASE), '--visits', '400', '--pairs', str(pairs),
                        '--seed', str(seed), '--name', name], self.out / f'cycle_{cycle:02d}' / phase)
        return directory

    def run(self):
        if (self.out / 'superseded.json').exists():
            raise RuntimeError('This pilot was superseded; resume pipelines/continuous_replay.sh instead')
        cfg = {'version': 1, 'cycles': 3, 'games_per_cycle': 2000, 'visits': 1200, 'fast_visits': 200,
               'full_prob': .06, 'pool_games': 100000, 'steps_per_cycle': 1000, 'batch': 64,
               'learning_rate': .0002, 'latest_fraction': .5, 'value_fraction': .5,
               'labels': 'terminal outcome, no search Q mixing', 'fixed_sha256': sha256(ROOT / FIXED),
               'initial_source_sha256': sha256(ROOT / BASE),
               'gate': '100 screening games; independent 200-game confirmation; Wilson and paired bootstrap lower bounds > .5',
               'sources': SOURCES}
        # JSON normalizes tuples to lists; compare canonical representations.
        cfg = json.loads(json.dumps(cfg))
        config_path = self.out / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != cfg:
            raise ValueError('Pilot configuration changed; use a new run name')
        atomic_json(config_path, cfg)
        champion = self.out / 'initial_champion.pt'
        if not champion.exists():
            tmp = champion.with_suffix('.tmp'); shutil.copy2(ROOT / BASE, tmp); os.replace(tmp, champion)
        if sha256(champion) != cfg['initial_source_sha256']: raise ValueError('Initial model changed')
        sources = [{'directory': str(ROOT / 'runs/experiments' / name), 'policy': policy} for name, policy in SOURCES]
        for source in sources:
            if not Path(source['directory']).is_dir(): raise FileNotFoundError(source['directory'])
        for cycle in range(1, 4):
            folder = self.out / f'cycle_{cycle:02d}'; folder.mkdir(exist_ok=True)
            schedule_path = self.out / 'search_schedule.json'
            schedule = json.loads(schedule_path.read_text()) if schedule_path.exists() else {}
            run, visits, seed = cycle_search(self.args.run, cycle, schedule)
            games = ROOT / 'runs/experiments' / run
            context = {'champion': str(champion), 'champion_sha256': sha256(champion),
                       'selfplay_run': run, 'seed': seed}
            if visits != 1200: context['search_visits'] = visits
            context_path = folder / 'context.json'
            if context_path.exists() and json.loads(context_path.read_text()) != context:
                raise ValueError('Frozen cycle producer changed')
            atomic_json(context_path, context)
            complete = folder / 'complete.json'
            self.update(cycle=cycle, phase='preparing', champion=str(champion.relative_to(ROOT)),
                        selfplay_run=run, train_dir=None, match=None,
                        visits=visits, fast_visits=200, full_prob=.06)
            if complete.exists():
                result = json.loads(complete.read_text())
                champion = Path(result['champion_after'])
                if sha256(champion) != result['champion_after_sha256']: raise ValueError('Promoted model changed')
                sources.append({'directory': str(games), 'policy': True})
                continue
            self.update(phase='selfplay', games_total=2000)
            summary = games / 'summary.json'
            saved_games = len(list(games.glob('game_*.json')))
            if not summary.exists() or saved_games != 2000:
                self.child([sys.executable, '-u', 'scripts/selfplay_parallel.py', '--run', run,
                            '--games', '2000', '--seed', str(context['seed']), '--model', str(champion),
                            '--reuse-tree', '--visits', str(visits), '--fast-visits', '200', '--full-prob', '.06',
                            '--fast-opening-sample'], folder / 'selfplay', ROOT / 'runs' / f'{run}.pid')
            if len(list(games.glob('game_*.json'))) != 2000: raise RuntimeError('Incomplete self-play batch')
            sources.append({'directory': str(games), 'policy': True})
            self.update(phase='pool_index', indexed_games=0)
            pool_path = folder / 'pool.json'
            validation_path = self.out / 'internal_validation.json'
            validation_reference = json.loads(validation_path.read_text()) if validation_path.exists() else None
            if not pool_path.exists():
                pool = build_pool(sources, ROOT / FIXED, ROOT / 'runs/replay_cache_v1', pool_path,
                                  latest=games, progress=lambda n: self.update(indexed_games=n),
                                  validation_reference=validation_reference)
            else:
                pool = json.loads(pool_path.read_text())
                if pool['fixed_manifest_sha256'] != sha256(ROOT / FIXED): raise ValueError('Holdout changed')
            if validation_reference is None:
                atomic_json(validation_path, pool['validation'])
            elif pool['validation'] != validation_reference:
                raise ValueError('Internal validation set changed')
            self.update(pool_games=len(pool['train']), latest_games=pool['latest_train_games'], phase='training')
            train = folder / 'train'
            self.update(train_dir=str(train.relative_to(ROOT)))
            self.child([sys.executable, '-u', 'scripts/train_replay.py', '--pool', str(pool_path),
                        '--init', str(champion), '--out', str(train), '--steps', '1000', '--batch', '64',
                        '--lr', '.0002', '--eval-every', '250', '--seed', str(20261120 + cycle)], folder / 'training')
            candidate = train / 'final.pt'
            self.update(phase='fixed_evaluation')
            fixed = folder / 'fixed_value.json'
            if not fixed.exists():
                self.child([sys.executable, '-u', 'scripts/eval_fixed_value.py', str(champion), str(candidate),
                            '--out', str(fixed)], folder / 'fixed_eval')
            tactics = ROOT / 'runs/diagnostics/t5_20261019/position.json'
            if tactics.exists() and not (folder / 't5.json').exists():
                self.update(phase='t5_diagnostic')
                self.child([sys.executable, '-u', 'scripts/eval_replay_tactics.py', str(champion), str(candidate),
                            '--position', str(tactics), '--out', str(folder / 't5.json')], folder / 't5')
            screen_dir = self.run_match(cycle, candidate, champion, 'screen', 50, 30300000 + cycle * 1000)
            screen = match_statistics([screen_dir])
            confirm = None
            if screen['games'] >= 95 and screen['win_rate'] >= .5:
                confirm_dir = self.run_match(cycle, candidate, champion, 'confirm', 100, 30300100 + cycle * 1000)
                confirm = match_statistics([confirm_dir])
            promoted = promotion_decision(screen, confirm)
            if promoted: champion = candidate
            result = {'cycle': cycle, 'promoted': promoted, 'screen': screen, 'confirmation': confirm,
                      'champion_after': str(champion), 'champion_after_sha256': sha256(champion),
                      'candidate_sha256': sha256(candidate)}
            atomic_json(complete, result)
            print(json.dumps(result), flush=True)
            self.update(phase='cycle_complete', last_result=result)
        self.update(phase='completed', champion=str(champion.relative_to(ROOT)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=DEFAULT_RUN)
    args = ap.parse_args()
    if not args.run or Path(args.run).name != args.run: ap.error('run must be a single folder name')
    os.chdir(ROOT)
    pilot = Pilot(args)
    def stop(signum, frame): raise KeyboardInterrupt('Replay pilot stopped')
    signal.signal(signal.SIGTERM, stop)
    with (ROOT / 'runs/replay_gpu.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another replay pilot owns the GPU lock')
        try:
            if shutil.disk_usage(ROOT).free < 15 * 1024**3: raise RuntimeError('Need 15 GiB free to start')
            active = subprocess.run(['pgrep', '-fl', r'scripts/(selfplay_parallel|train_search|train_replay|match_parallel|eval_fixed_value|eval_replay_tactics)\.py'],
                                    capture_output=True, text=True)
            if active.stdout.strip():
                raise RuntimeError('Another GPU batch job is still active; stop it before resuming: ' + active.stdout.strip())
            pilot.run()
        except BaseException as exc:
            pilot.update(phase='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc), child_pid=None)
            raise


if __name__ == '__main__':
    main()
