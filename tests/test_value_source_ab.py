import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location('value_source_ab', Path(__file__).resolve().parents[1] / 'scripts/value_source_ab.py')
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


def test_heldout_partitions_keep_entire_games_together():
    rows = np.array([(s, g, p) for s in range(3) for g in range(100) for p in (10, 80, 200)])
    val, report = ab.heldout_partition(rows), ab.heldout_partition(rows, True)
    games = lambda a: set(map(tuple, a[:, :2]))
    assert games(val).isdisjoint(games(report))
    assert games(val) | games(report) == games(rows)
    assert len(val) + len(report) == len(rows)


def test_shared_evaluation_rows_are_deterministic_unique_and_stratified():
    rows = np.array([(s, g, p) for s in range(3) for g in range(40) for p in (10, 40, 100, 200)])
    selected = ab.select_rows(rows, 120, 7)
    assert np.array_equal(selected, ab.select_rows(rows, 120, 7))
    assert len(set(map(tuple, selected))) == len(selected) == 120
    for s in range(3):
        for lo, hi, _ in ab.PHASES:
            assert ((selected[:, 0] == s) & (selected[:, 2] >= lo) & (selected[:, 2] < hi)).sum() == 10


def test_checkpoint_selection_uses_equal_sources_not_training_population():
    metrics = {'gen': {'mse': .2}, 'rl': {'mse': .8}, 'human': {'mse': .5}, 'all': {'mse': .9}}
    assert ab.selection_score(metrics) == .5


def test_evaluation_masks_unknown_results_and_keeps_source_phase_metrics():
    class Source:
        name = 'gen'
    class Data:
        sources = [Source()]
        def batch_from_index(self, rows):
            ids = rows[:, 1]
            f = np.zeros((len(rows), 1, 1, 1), np.float32)
            f[:, 0, 0, 0] = np.array([1., -1., .7])[ids]
            return f, np.array([1., -1., 0.], np.float32)[ids], None
    class Net(torch.nn.Module):
        def forward(self, x):
            return None, x[:, 0, 0, 0]
    rows = np.array([[0, 0, 100], [0, 1, 200], [0, 2, 100]])
    result = ab.evaluate(Net(), Data(), rows, torch.device('cpu'), 2)
    assert result['all']['n'] == 2
    assert result['all']['mse'] == 0
    assert result['all']['winner_acc'] == 1
    assert result['all']['ece'] == 0
    assert result['gen_middle']['n'] == 1


def test_eta_accounts_for_sequential_wait_and_unstarted_arm():
    state = {'steps': 200, 'mixed': {'step': 100}}
    estimates, total = ab.remaining_estimates(state, {'mixed': .2, 'generated': None})
    assert estimates['mixed']['remaining'] == 20
    assert estimates['generated']['remaining'] == 40
    assert estimates['generated']['finish_after'] == total == 60
    assert estimates['generated']['borrowed_speed']
    estimates, total = ab.remaining_estimates({'steps': 200}, {'mixed': None, 'generated': None})
    assert total is None
    assert ab.duration(3661) == '01:01:01'


def test_recent_speed_ignores_incomplete_records_and_uses_effective_batch(tmp_path):
    p = tmp_path / 'metrics.jsonl'
    p.write_text('{"step": 100, "samples_per_s": 1280}\n'
                 '{"step": 200, "samples_per_s": 640}\n'
                 '{"step": 300, "samples_per_s": 0}\n{"step":')
    assert ab.recent_step_seconds(p, 256) == .3
