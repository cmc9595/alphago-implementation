import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_search import SelfPlay
import train_search
from alphago.models import GoNet, NetConfig


def test_concatenation_releases_chunks_and_preserves_order():
    arrays = [np.ones((2, 3), np.uint8), np.zeros((1, 3), np.uint8)]
    expected = np.concatenate(arrays)
    actual = train_search.concatenate_releasing(arrays)
    np.testing.assert_array_equal(actual, expected)
    assert arrays == []


def test_holdout_excludes_duplicate_moves_from_policy_and_value_sources(tmp_path):
    policy, value = tmp_path / 'policy', tmp_path / 'value'
    policy.mkdir(); value.mkdir()
    heldout = np.array([60, 300], dtype='<i2')
    digest = hashlib.sha256(heldout.tobytes()).hexdigest()
    for folder in (policy, value):
        # No other fields: exclusion must happen before loading any targets.
        np.savez(folder / 'game_9999.npz', moves=heldout)
        for i in range(4):
            np.savez(folder / f'game_{i:04d}.npz', moves=np.array([i, 20 + i], np.int16),
                     capped=False, judged=True, score=1., seed=i,
                     train_mask=np.ones(2, bool), features=np.zeros((2, 34, 19, 19), np.uint8),
                     policy=np.full((2, 362), 1 / 362, np.float32),
                     search_values=np.zeros(2, np.float32), ownership=np.ones(361, np.int8))
    sp = SelfPlay([policy], .5, 1, val_mod=2, value_only_dirs=[value], exclude_hashes={digest})
    assert sp.excluded == 2
    assert sp.games == 8
    assert len(sp.z) == 16
    assert sp.policy_mask.sum() == 8


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='trainer requires MPS')
@pytest.mark.parametrize('global_pool', [False, True])
def test_selfplay_only_training_never_loads_anchor_data(tmp_path, monkeypatch, global_pool):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train_search, 'ROOT', tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled anchor data must not be loaded')
    monkeypatch.setattr(train_search, 'GameData', forbidden)
    monkeypatch.setattr(train_search, 'ValueData', forbidden)
    monkeypatch.setattr(train_search, 'load_gen', forbidden)
    monkeypatch.setattr(train_search, 'memory_guard', lambda: None)
    cfg = NetConfig(blocks=3 if global_pool else 1, channels=4, ownership=True, global_pool=global_pool)
    net = GoNet(cfg)
    torch.save({'model': net.state_dict(), 'config': vars(cfg)}, tmp_path / 'init.pt')
    games = tmp_path / 'games'; games.mkdir()
    for i in range(20):
        np.savez(games / f'game_{i:04d}.npz', moves=np.array([i, i + 20], np.int16),
                 capped=False, judged=True, score=1. if i % 2 else -1., seed=i,
                 train_mask=np.ones(2, bool), features=np.zeros((2, 34, 19, 19), np.uint8),
                 policy=np.full((2, 362), 1 / 362, np.float32),
                 search_values=np.zeros(2, np.float32), ownership=np.ones(361, np.int8))
    monkeypatch.setattr(sys, 'argv', ['train_search.py', '--selfplay', str(games),
                        '--init', 'init.pt', '--run', 'smoke', '--steps', '2', '--batch', '4',
                        '--human-batch', '0', '--gen-batch', '0', '--eval-every', '1', '--value-fraction', '1'])
    train_search.main()
    ck = torch.load(tmp_path / 'checkpoints/smoke/final.pt', map_location='cpu', weights_only=False)
    assert len(ck['history']) == 2
    assert all(np.isfinite(row['loss']) for row in ck['history'])
    assert 'human_policy_acc' not in ck['history'][-1]
    assert 'gen_value_mse' not in ck['history'][-1]
    assert ck['config']['global_pool'] == global_pool
