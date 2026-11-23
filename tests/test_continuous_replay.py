import argparse
import json
from pathlib import Path
import sys

import pytest

from alphago.replay import atomic_json, sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import continuous_replay as C
import replay_loop


def test_learner_advances_even_if_producer_does_not():
    assert C.advance_models('producer', 'candidate', False) == ('producer', 'candidate')
    assert C.advance_models('producer', 'candidate', True) == ('candidate', 'candidate')


def test_generation_change_is_prospective_and_keeps_original_contexts():
    schedule = {'3': {'visits': 1000, 'full_prob': .25, 'seed': 22350000}}
    assert C.generation_settings('batch2', 22200000, 2, schedule) == ('batch2', 22200000, 1600, .06)
    assert C.generation_settings('batch3', 22300000, 3, schedule) == ('batch3_v1000_p25', 22350000, 1000, .25)
    for bad in [{'full_prob': 0}, {'full_prob': 1.1}, {'visits': 100}, {'seed': -1}, {'unsupported': 1}]:
        with pytest.raises(ValueError): C.generation_settings('batch', 1, 3, {'3': bad})


def test_value_supervision_override_preserves_previous_cycles():
    schedule = {'3': {'value_fraction': 1.0}}
    assert C.training_settings(1, schedule) == .5
    assert C.training_settings(2, schedule) == .5
    assert C.training_settings(3, schedule) == 1.
    for bad in [{'value_fraction': 0}, {'value_fraction': 1.1}, {'unknown': 1}]:
        with pytest.raises(ValueError): C.training_settings(3, {'3': bad})


def test_training_budget_override_is_prospective_and_validated():
    schedule = {'3': {'value_fraction': 1., 'batch': 512, 'steps': 250, 'eval_every': 50}}
    assert C.training_budget(2, schedule) == {'batch': 64, 'steps': 1000, 'eval_every': 250}
    assert C.training_budget(3, schedule) == {'batch': 512, 'steps': 250, 'eval_every': 50}
    for bad in [{'batch': 0}, {'steps': -1}, {'eval_every': 1.5}]:
        with pytest.raises(ValueError): C.training_budget(3, {'3': bad})


def test_extending_batch_rejects_search_or_producer_changes(tmp_path):
    producer = tmp_path / 'model.pt'; producer.write_bytes(b'weights')
    args = {'visits': 1600, 'fast_visits': 200, 'full_prob': .06, 'seed': 21150000,
            'reuse_tree': True, 'fast_opening_sample': True, 'resign': False, 'max_moves': 600,
            'parallel_games': 32, 'games': 2000, 'model': str(producer)}
    C.validate_extension({'args': args}, 21150000, sha256(producer))
    with pytest.raises(ValueError):
        C.validate_extension({'args': {**args, 'visits': 1200}}, 21150000, sha256(producer))
    with pytest.raises(ValueError):
        C.validate_extension({'args': args}, 21150000, 'wrong model hash')


def test_three_failed_gates_keep_producer_but_continue_learner(tmp_path, monkeypatch):
    monkeypatch.setattr(C, 'ROOT', tmp_path)
    monkeypatch.setattr(replay_loop, 'ROOT', tmp_path)
    old = tmp_path / 'runs/replay/old'; old.mkdir(parents=True)
    base = tmp_path / C.BASE; base.parent.mkdir(parents=True); base.write_bytes(b'producer')
    (old / 'initial_champion.pt').write_bytes(b'producer')
    prior = old / 'cycle_01'; (prior / 'train').mkdir(parents=True)
    previous = prior / 'train/final.pt'; previous.write_bytes(b'prior failed learner')
    atomic_json(prior / 'context.json', {'selfplay_run': 'old_games'})
    atomic_json(prior / 'complete.json', {'champion_after': str(old / 'initial_champion.pt'),
                'champion_after_sha256': sha256(base), 'candidate_sha256': sha256(previous)})
    fixed = tmp_path / C.FIXED
    atomic_json(fixed, {'test_games': [{'sha256_moves': 'fixed'}]})
    reference = [{'hash': 'monitor'}]
    atomic_json(old / 'internal_validation.json', reference)
    for name, _ in C.SOURCES:
        (tmp_path / 'runs/experiments' / name).mkdir(parents=True)
    (tmp_path / 'runs/experiments/old_games').mkdir()
    monkeypatch.setattr(C, 'game_count', lambda p: 5000 if (p / 'summary.json').exists() else 0)
    def pool(sources, fixed, cache, output, **kwargs):
        body = {'train': [{}] * 100, 'latest_train_games': 10, 'validation': reference,
                'fixed_manifest_sha256': sha256(fixed)}
        atomic_json(output, body)
        return body
    monkeypatch.setattr(C, 'build_pool', pool)
    monkeypatch.setattr(C, 'match_statistics', lambda dirs: {'games': 100, 'win_rate': .49})
    training_inputs, generation_inputs = [], []
    class FakePilot(C.ContinuousPilot):
        def child(self, cmd, stage, pidfile=None):
            if 'scripts/selfplay_parallel.py' in cmd:
                generation_inputs.append(cmd[cmd.index('--model') + 1])
                directory = tmp_path / 'runs/experiments' / cmd[cmd.index('--run') + 1]
                atomic_json(directory / 'summary.json', {})
            elif 'scripts/train_replay.py' in cmd:
                training_inputs.append(cmd[cmd.index('--init') + 1])
                assert cmd[cmd.index('--sampling') + 1] == 'uniform'
                if len(training_inputs) == 3:
                    assert cmd[cmd.index('--value-fraction') + 1] == '1.0'
                    assert cmd[cmd.index('--batch') + 1] == '512'
                    assert cmd[cmd.index('--steps') + 1] == '250'
                    assert cmd[cmd.index('--eval-every') + 1] == '50'
                else:
                    assert '--value-fraction' not in cmd
                out = Path(cmd[cmd.index('--out') + 1]); out.mkdir(parents=True)
                (out / 'final.pt').write_bytes(f'learner {len(training_inputs)}'.encode())
            elif 'scripts/eval_fixed_value.py' in cmd:
                atomic_json(Path(cmd[cmd.index('--out') + 1]), {})
            else: raise AssertionError(cmd)
        def run_match(self, *args):
            return tmp_path / 'dummy_match'
    args = argparse.Namespace(run='new', import_pilot='old', first_batch='first', first_seed=21150000)
    pilot = FakePilot(args)
    atomic_json(pilot.out / 'training_schedule.json', {'3': {'value_fraction': 1.0, 'batch': 512, 'steps': 250, 'eval_every': 50}})
    pilot.run()
    assert len(training_inputs) == len(generation_inputs) == 3
    assert len(set(generation_inputs)) == 1
    assert training_inputs[1].endswith('cycle_01/train/final.pt')
    assert training_inputs[2].endswith('cycle_02/train/final.pt')
    for cycle in range(1, 4):
        result = json.loads((pilot.out / f'cycle_{cycle:02d}/complete.json').read_text())
        assert not result['promoted'] and result['learner_after'] != result['champion_after']
    # Completed runs are safely reusable: no repeated generation or training.
    pilot.run()
    assert len(training_inputs) == len(generation_inputs) == 3
