"""Uniform replay with separate cumulative learner and gated self-play producer.

Runs three 5,000-game cycles. May extend the interrupted pilot's compatible
third batch; all original artifacts remain intact.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from alphago.replay import atomic_json, build_pool, sha256
from replay_loop import BASE, FIXED, ROOT, SOURCES, Pilot, match_statistics, promotion_decision

DEFAULT_RUN = 'replay_continuous_1600v_p06_5000_20261121'
OLD_RUN = 'replay_pilot_1200v_p06_20261120'


def game_count(directory):
    return len(list(Path(directory).glob('game_*.json')))


def advance_models(producer, candidate, promoted):
    return (candidate if promoted else producer), candidate


def generation_settings(run, seed, cycle, schedule):
    """Prospective overrides use a separate generation directory and frozen context."""
    override = schedule.get(str(cycle), {})
    if set(override) - {'visits', 'full_prob', 'seed'}: raise ValueError('Unknown generation override')
    visits = override.get('visits', 1600)
    probability = override.get('full_prob', .06)
    seed = override.get('seed', seed)
    if type(visits) is not int or visits < 200 or type(seed) is not int or seed < 0:
        raise ValueError('Invalid generation visits/seed')
    if not isinstance(probability, (int, float)) or not 0 < probability <= 1:
        raise ValueError('Invalid full-search probability')
    if visits != 1600 or probability != .06:
        percentage = format(probability * 100, 'g').replace('.', 'd')
        run += f'_v{visits}_p{percentage}'
    return run, seed, visits, probability


def training_settings(cycle, schedule):
    override = schedule.get(str(cycle), {})
    if set(override) - {'value_fraction', 'batch', 'steps', 'eval_every'}: raise ValueError('Unknown training override')
    fraction = override.get('value_fraction', .5)
    if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError('Invalid value supervision fraction')
    return float(fraction)


def training_budget(cycle, schedule):
    training_settings(cycle, schedule)
    override = schedule.get(str(cycle), {})
    budget = {k: override.get(k, default) for k, default in [('batch', 64), ('steps', 1000), ('eval_every', 250)]}
    if any(type(v) is not int or v <= 0 for v in budget.values()):
        raise ValueError('Training batch, steps and evaluation interval must be positive integers')
    return budget


def frozen_copy(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        if sha256(source) != sha256(target): raise ValueError(f'Frozen checkpoint changed: {target}')
    else:
        tmp = target.with_suffix('.tmp'); shutil.copy2(source, tmp); os.replace(tmp, target)
    return target


def import_lineage(old):
    sources = []
    producer, learner = old / 'initial_champion.pt', None
    for cycle in range(1, 4):
        folder = old / f'cycle_{cycle:02d}'
        complete = folder / 'complete.json'
        if not complete.exists(): break
        result = json.loads(complete.read_text())
        producer = Path(result['champion_after'])
        if sha256(producer) != result['champion_after_sha256']: raise ValueError('Imported producer changed')
        candidate = folder / 'train/final.pt'
        if sha256(candidate) != result['candidate_sha256']: raise ValueError('Imported learner changed')
        learner = candidate
        context = json.loads((folder / 'context.json').read_text())
        sources.append({'directory': str(ROOT / 'runs/experiments' / context['selfplay_run']), 'policy': True})
    if learner is None: raise ValueError('No completed learner to continue')
    return producer, learner, sources


def validate_extension(manifest, seed, model_sha, target=5000):
    a = manifest['args']
    expected = {'visits': 1600, 'fast_visits': 200, 'full_prob': .06, 'seed': seed,
                'reuse_tree': True, 'fast_opening_sample': True, 'resign': False,
                'max_moves': 600, 'parallel_games': 32}
    if any(a.get(k) != v for k, v in expected.items()) or a['games'] not in (2000, target):
        raise ValueError('Existing first batch has incompatible generation settings')
    if sha256(ROOT / a['model']) != model_sha: raise ValueError('Existing first batch uses a different producer')


class ContinuousPilot(Pilot):
    def run(self):
        imported = ROOT / 'runs/replay' / self.args.import_pilot
        producer_source, learner_source, extra_sources = import_lineage(imported)
        reference = imported / 'internal_validation.json'
        cfg = {'version': 2, 'cycles': 3, 'games_per_cycle': 5000, 'visits': 1600, 'fast_visits': 200,
               'full_prob': .06, 'pool_games': 100000, 'steps_per_cycle': 1000, 'batch': 64,
               'learning_rate': .0002, 'sampling': 'uniform_games', 'value_fraction': .5, 'search_mix': 0,
               'learner_continues_without_promotion': True, 'fixed_sha256': sha256(ROOT / FIXED),
               'import_pilot': self.args.import_pilot, 'initial_producer_sha256': sha256(producer_source),
               'initial_learner_sha256': sha256(learner_source), 'validation_sha256': sha256(reference),
               'first_batch': self.args.first_batch, 'first_seed': self.args.first_seed,
               'extra_sources': extra_sources, 'sources': SOURCES,
               'gate': '100 screening, independent 200 confirmation, unchanged pilot thresholds'}
        cfg = json.loads(json.dumps(cfg))
        config = self.out / 'config.json'
        if config.exists() and json.loads(config.read_text()) != cfg: raise ValueError('Continuous configuration changed')
        atomic_json(config, cfg)
        producer = frozen_copy(producer_source, self.out / 'initial_producer.pt')
        learner = frozen_copy(learner_source, self.out / 'initial_learner.pt')
        fixed_reference = self.out / 'internal_validation.json'
        if fixed_reference.exists() and sha256(fixed_reference) != cfg['validation_sha256']:
            raise ValueError('Internal validation reference changed')
        if not fixed_reference.exists(): frozen_copy(reference, fixed_reference)
        monitored = json.loads(fixed_reference.read_text())
        sources = [{'directory': str(ROOT / 'runs/experiments' / name), 'policy': policy} for name, policy in SOURCES]
        sources.extend(extra_sources)
        for source in sources:
            if not Path(source['directory']).is_dir(): raise FileNotFoundError(source['directory'])
        self.update(mode='continuous', sampling='uniform_games', games_total=5000,
                    visits=1600, fast_visits=200, full_prob=.06)
        for cycle in range(1, 4):
            folder = self.out / f'cycle_{cycle:02d}'; folder.mkdir(exist_ok=True)
            run = self.args.first_batch if cycle == 1 else f'{self.args.run}_selfplay_c{cycle:02d}'
            seed = self.args.first_seed if cycle == 1 else 22000000 + cycle * 100000
            schedule_path = self.out / 'generation_schedule.json'
            schedule = json.loads(schedule_path.read_text()) if schedule_path.exists() else {}
            run, seed, visits, full_prob = generation_settings(run, seed, cycle, schedule)
            training_schedule_path = self.out / 'training_schedule.json'
            training_schedule = json.loads(training_schedule_path.read_text()) if training_schedule_path.exists() else {}
            value_fraction = training_settings(cycle, training_schedule)
            budget = training_budget(cycle, training_schedule)
            games = ROOT / 'runs/experiments' / run
            context = {'producer': str(producer), 'producer_sha256': sha256(producer),
                       'learner_input': str(learner), 'learner_input_sha256': sha256(learner),
                       'selfplay_run': run, 'seed': seed, 'games': 5000, 'visits': visits}
            if visits != 1600 or full_prob != .06: context['full_prob'] = full_prob
            context_path = folder / 'context.json'
            if context_path.exists() and json.loads(context_path.read_text()) != context:
                raise ValueError('Frozen producer/learner context changed')
            atomic_json(context_path, context)
            self.update(cycle=cycle, phase='preparing', champion=str(producer.relative_to(ROOT)),
                        learner=str(learner.relative_to(ROOT)), selfplay_run=run, train_dir=None, match=None,
                        visits=visits, fast_visits=200, full_prob=full_prob, value_fraction=value_fraction,
                        batch=budget['batch'], train_steps=budget['steps'], eval_every=budget['eval_every'])
            complete = folder / 'complete.json'
            if complete.exists():
                result = json.loads(complete.read_text())
                producer, learner = Path(result['champion_after']), Path(result['learner_after'])
                if sha256(producer) != result['champion_after_sha256'] or sha256(learner) != result['learner_after_sha256']:
                    raise ValueError('Completed checkpoint changed')
                sources.append({'directory': str(games), 'policy': True})
                continue
            if cycle == 1 and (games / 'manifest.json').exists():
                raw = json.loads((games / 'manifest.json').read_text())
                validate_extension(raw, seed, context['producer_sha256'])
                before = folder / 'first_batch_manifest_before_extension.json'
                if not before.exists(): atomic_json(before, raw)
            saved = game_count(games)
            if saved > 5000: raise ValueError('Too many games in batch')
            self.update(phase='selfplay')
            if saved != 5000 or not (games / 'summary.json').exists():
                self.child([sys.executable, '-u', 'scripts/selfplay_parallel.py', '--run', run,
                            '--games', '5000', '--seed', str(seed), '--model', str(producer),
                            '--reuse-tree', '--visits', str(visits), '--fast-visits', '200', '--full-prob',
                            '.06' if full_prob == .06 else str(full_prob),
                            '--fast-opening-sample'], folder / 'selfplay', ROOT / 'runs' / f'{run}.pid')
            if game_count(games) != 5000: raise RuntimeError('Incomplete generation')
            sources.append({'directory': str(games), 'policy': True})
            pool_path = folder / 'pool.json'
            self.update(phase='pool_index', indexed_games=0)
            if not pool_path.exists():
                pool = build_pool(sources, ROOT / FIXED, ROOT / 'runs/replay_cache_v1', pool_path,
                                  latest=games, validation_reference=monitored,
                                  progress=lambda n: self.update(indexed_games=n))
            else:
                pool = json.loads(pool_path.read_text())
            if pool['validation'] != monitored or pool['fixed_manifest_sha256'] != cfg['fixed_sha256']:
                raise ValueError('Evaluation separation changed')
            train = folder / 'train'
            self.update(phase='training', train_dir=str(train.relative_to(ROOT)),
                        pool_games=len(pool['train']), latest_games=pool['latest_train_games'])
            training_context = {'init_sha256': sha256(learner), 'pool_sha256': sha256(pool_path),
                                'value_fraction': value_fraction, 'steps': budget['steps'], 'batch': budget['batch'],
                                'sampling': 'uniform_games'}
            if budget['eval_every'] != 250: training_context['eval_every'] = budget['eval_every']
            training_context_path = folder / 'training_context.json'
            if training_context_path.exists() and json.loads(training_context_path.read_text()) != training_context:
                raise ValueError('Frozen training configuration changed')
            atomic_json(training_context_path, training_context)
            command = [sys.executable, '-u', 'scripts/train_replay.py', '--pool', str(pool_path),
                       '--init', str(learner), '--out', str(train), '--steps', str(budget['steps']), '--batch', str(budget['batch']),
                       '--lr', '.0002', '--eval-every', str(budget['eval_every']), '--seed', str(20261121 + cycle), '--sampling', 'uniform']
            if value_fraction != .5: command += ['--value-fraction', str(value_fraction)]
            self.child(command, folder / 'training')
            candidate = train / 'final.pt'
            self.update(phase='fixed_evaluation')
            if not (folder / 'fixed_value.json').exists():
                self.child([sys.executable, '-u', 'scripts/eval_fixed_value.py', str(producer), str(candidate),
                            '--out', str(folder / 'fixed_value.json')], folder / 'fixed_eval')
            tactics = ROOT / 'runs/diagnostics/t5_20261019/position.json'
            if tactics.exists() and not (folder / 't5.json').exists():
                self.update(phase='t5_diagnostic')
                self.child([sys.executable, '-u', 'scripts/eval_replay_tactics.py', str(producer), str(candidate),
                            '--position', str(tactics), '--out', str(folder / 't5.json')], folder / 't5')
            screen_dir = self.run_match(cycle, candidate, producer, 'screen', 50, 30400000 + cycle * 1000)
            screen = match_statistics([screen_dir]); confirm = None
            if screen['games'] >= 95 and screen['win_rate'] >= .5:
                confirm_dir = self.run_match(cycle, candidate, producer, 'confirm', 100, 30400100 + cycle * 1000)
                confirm = match_statistics([confirm_dir])
            promoted = promotion_decision(screen, confirm)
            # Training ALWAYS continues from the latest candidate, independently of gating.
            producer, learner = advance_models(producer, candidate, promoted)
            result = {'cycle': cycle, 'promoted': promoted, 'screen': screen, 'confirmation': confirm,
                      'champion_after': str(producer), 'champion_after_sha256': sha256(producer),
                      'learner_after': str(learner), 'learner_after_sha256': sha256(learner),
                      'candidate_sha256': sha256(candidate)}
            atomic_json(complete, result)
            print(json.dumps(result), flush=True)
            self.update(phase='cycle_complete', last_result=result)
        self.update(phase='completed', champion=str(producer.relative_to(ROOT)), learner=str(learner.relative_to(ROOT)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=DEFAULT_RUN)
    ap.add_argument('--import-pilot', default=OLD_RUN)
    ap.add_argument('--first-batch', default=OLD_RUN + '_selfplay_c03_v1600')
    ap.add_argument('--first-seed', type=int, default=21150000)
    args = ap.parse_args()
    if any(not s or Path(s).name != s for s in (args.run, args.import_pilot, args.first_batch)):
        ap.error('Run identifiers must be single folder names')
    os.chdir(ROOT)
    pilot = ContinuousPilot(args)
    def stop(signum, frame): raise KeyboardInterrupt('Continuous replay stopped')
    signal.signal(signal.SIGTERM, stop)
    with (ROOT / 'runs/replay_gpu.lock').open('a+') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another replay controller owns the GPU lock')
        try:
            if shutil.disk_usage(ROOT).free < 15 * 1024**3: raise RuntimeError('Need 15 GiB free to start')
            active = subprocess.run(['pgrep', '-fl', r'scripts/(selfplay_parallel|train_search|train_replay|match_parallel|eval_fixed_value|eval_replay_tactics)\.py'], capture_output=True, text=True)
            if active.stdout.strip(): raise RuntimeError('Another GPU batch is active: ' + active.stdout.strip())
            pilot.run()
        except BaseException as exc:
            pilot.update(phase='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc), child_pid=None)
            raise


if __name__ == '__main__':
    main()
