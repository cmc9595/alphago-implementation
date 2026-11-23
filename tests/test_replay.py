import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from alphago.models import GoNet, NetConfig
from alphago.replay import ReplaySampler, atomic_json, build_pool, cache_game, move_hash, validation_game

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from replay_loop import cycle_search, match_statistics, promotion_decision
import train_replay


def write_game(folder, i, positions=2, *, capped=False):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f'game_{i:04d}.npz'
    moves = np.array([i % 361, (i * 7 + 3) % 361] + [361] * positions, np.int16)
    mask = np.zeros(len(moves), bool); mask[:positions] = True
    np.savez(p, moves=moves, train_mask=mask, capped=capped, judged=True, score=7.5,
             features=np.zeros((len(moves), 34, 19, 19), np.uint8),
             policy=np.full((len(moves), 362), 1 / 362, np.float32), ownership=np.ones(361, np.int8))
    p.with_suffix('.json').write_text('{}')
    return p


def test_cache_is_compact_incremental_and_preserves_side_to_move(tmp_path):
    path = write_game(tmp_path / 'source', 1, positions=3)
    rec = cache_game(path, tmp_path / 'cache')
    target = Path(rec['cache']); before = target.stat().st_mtime_ns
    with np.load(target) as d:
        assert d['x'].shape == (3, 5, 19, 19)
        assert d['z'].tolist() == [1., -1., 1.]
        assert d['own'][:, 0, 0].tolist() == [1, -1, 1]
    assert cache_game(path, tmp_path / 'cache') == rec
    assert target.stat().st_mtime_ns == before
    assert cache_game(write_game(tmp_path / 'source', 2, capped=True), tmp_path / 'cache')['reason']
    assert cache_game(write_game(tmp_path / 'source', 3, positions=0), tmp_path / 'cache')['reason'] == 'no_full_search_positions'


def test_pool_excludes_holdout_and_deduplicates_across_sources(tmp_path):
    old, new = tmp_path / 'old', tmp_path / 'new'
    for i in range(100): write_game(old, i)
    for i in range(80, 140): write_game(new, i)
    with np.load(old / 'game_0001.npz') as d: excluded = move_hash(d['moves'])
    fixed = tmp_path / 'fixed.json'; atomic_json(fixed, {'test_games': [{'sha256_moves': excluded}]})
    pool = build_pool([{'directory': str(old), 'policy': False}, {'directory': str(new), 'policy': True}],
                      fixed, tmp_path / 'cache', tmp_path / 'pool.json', max_games=100, latest=new)
    hashes = [r['hash'] for r in pool['train']]
    assert len(hashes) == 100 and len(set(hashes)) == 100
    assert excluded not in hashes
    assert not set(hashes).intersection(r['hash'] for r in pool['validation'])
    assert all(not validation_game(h) for h in hashes)
    assert pool['rejected']['duplicate'] == 20
    assert pool['rejected']['fixed_holdout'] == 1
    assert pool['latest_train_games'] > 0
    again = build_pool([{'directory': str(old), 'policy': False}, {'directory': str(new), 'policy': True}],
                       fixed, tmp_path / 'cache', tmp_path / 'pool2.json', max_games=100, latest=new)
    assert pool == again
    for i in range(140, 160): write_game(new, i)
    updated = build_pool([{'directory': str(old), 'policy': False}, {'directory': str(new), 'policy': True}],
                         fixed, tmp_path / 'cache', tmp_path / 'pool3.json', max_games=100, latest=new,
                         validation_reference=pool['validation'])
    assert updated['validation'] == pool['validation']


def tiny_pool(tmp_path):
    rows = []
    for i in range(22):
        r = cache_game(write_game(tmp_path / 'source', i, positions=2 if i % 2 else 8), tmp_path / 'cache')
        rows.append({**r, 'latest': i < 10, 'policy': True})
    return {'train': rows[:-2], 'validation': rows[-2:], 'fixed_manifest_sha256': 'test'}


def test_sampler_is_game_uniform_stratified_distinct_and_bounded(tmp_path):
    pool = tiny_pool(tmp_path)
    sampler = ReplaySampler(pool, cache_games=3)
    a = sampler.sample(8, np.random.default_rng(7), augment=False, value_fraction=1)
    b = sampler.sample(8, np.random.default_rng(7), augment=False, value_fraction=1)
    for x, y in zip(a[:-1], b[:-1]): np.testing.assert_array_equal(x, y)
    assert a[-1] == b[-1]
    assert a[5].all()
    identities = a[-1]
    latest = {r['hash'] for r in sampler.latest}
    assert sum(h in latest for h, _ in identities) == 4
    assert len({h for h, _ in identities}) == 8
    assert len(sampler.cache) <= 3
    counts = {r['hash']: 0 for r in pool['train']}
    rng = np.random.default_rng(4)
    for _ in range(200):
        for h, _ in sampler.sample(8, rng, augment=False)[-1]: counts[h] += 1
    # Eight-position games must not be sampled 4x as often as two-position games.
    short = sum(counts[r['hash']] for r in pool['train'] if r['positions'] == 2)
    long = sum(counts[r['hash']] for r in pool['train'] if r['positions'] == 8)
    assert .8 < short / long < 1.2
    with pytest.raises(ValueError, match='leaked'):
        ReplaySampler({**pool, 'validation': [pool['train'][0]]})


def test_uniform_sampling_has_no_latest_oversampling(tmp_path):
    pool = tiny_pool(tmp_path)
    for i, row in enumerate(pool['train']): row['latest'] = i < 2
    sampler = ReplaySampler(pool)
    latest = {r['hash'] for r in sampler.latest}
    rng = np.random.default_rng(7)
    latest_draws = total = 0
    for _ in range(200):
        batch = sampler.sample(8, rng, uniform=True, augment=False)[-1]
        assert len(set(h for h, _ in batch)) == 8
        latest_draws += sum(h in latest for h, _ in batch)
        total += len(batch)
    assert .07 < latest_draws / total < .13  # 2 / 20, NOT 50%.


def test_gate_requires_independent_confirmation_and_enough_pairs(tmp_path):
    directory = tmp_path / 'match'; directory.mkdir()
    for i in range(200):
        atomic_json(directory / f'game{i:03d}.json', {'pair': i // 2, 'a_score': 1. if i < 140 else 0., 'capped': False})
    stats = match_statistics([directory])
    screen = {'games': 100, 'win_rate': .55}
    assert stats['games'] == 200 and stats['complete_pairs'] == 100
    assert promotion_decision(screen, stats)
    assert not promotion_decision(screen, None)
    assert not promotion_decision({'games': 100, 'win_rate': .49}, stats)
    assert not promotion_decision(screen, {**stats, 'complete_pairs': 94})


def test_search_override_is_prospective_and_separates_batches():
    schedule = {'2': {'visits': 1600, 'seed': 21050000}, '3': {'visits': 1600, 'seed': 21150000}}
    assert cycle_search('pilot', 1, schedule) == ('pilot_selfplay_c01', 1200, 20900000)
    assert cycle_search('pilot', 2, schedule) == ('pilot_selfplay_c02_v1600', 1600, 21050000)
    assert cycle_search('pilot', 3, schedule) == ('pilot_selfplay_c03_v1600', 1600, 21150000)
    with pytest.raises(ValueError): cycle_search('pilot', 2, {'2': {'visits': -1}})


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS trainer')
@pytest.mark.parametrize('sampling,value_fraction', [('stratified', .5), ('uniform', .5), ('uniform', 1.)])
def test_trainer_resumes_optimizer_and_rng_without_anchor_data(tmp_path, monkeypatch, sampling, value_fraction):
    pool = tiny_pool(tmp_path)
    # Keep validation identities disjoint from training in this fixture too.
    pool['validation'] = pool['train'][-2:]; pool['train'] = pool['train'][:-2]
    path = tmp_path / 'pool.json'; atomic_json(path, pool)
    torch.manual_seed(9)
    net = GoNet(NetConfig(blocks=3, channels=4, ownership=True, global_pool=True))
    init = tmp_path / 'init.pt'; torch.save({'model': net.state_dict(), 'config': vars(net.cfg)}, init)
    def run(out, source=init):
        monkeypatch.setattr(sys, 'argv', ['train_replay.py', '--pool', str(path), '--init', str(source),
                            '--out', str(out), '--steps', '2', '--batch', '4', '--eval-every', '1', '--seed', '8',
                            '--sampling', sampling, '--value-fraction', str(value_fraction)])
        train_replay.main()
    monkeypatch.setattr(train_replay, 'memory_guard', lambda: None)
    uninterrupted = tmp_path / 'full'; run(uninterrupted)
    calls = []
    def interrupt():
        calls.append(1)
        if len(calls) == 2: raise RuntimeError('simulated interruption')
    monkeypatch.setattr(train_replay, 'memory_guard', interrupt)
    resumed = tmp_path / 'resume'
    with pytest.raises(RuntimeError, match='simulated interruption'): run(resumed)
    assert (resumed / 'last.pt').exists() and not (resumed / 'final.pt').exists()
    monkeypatch.setattr(train_replay, 'memory_guard', lambda: None)
    run(resumed)
    full = torch.load(uninterrupted / 'final.pt', map_location='cpu', weights_only=False)
    after = torch.load(resumed / 'final.pt', map_location='cpu', weights_only=False)
    assert after['step'] == 2 and after['game_exposures'] == full['game_exposures']
    assert after['numpy_rng'] == full['numpy_rng']
    for key in full['model']:
        torch.testing.assert_close(full['model'][key], after['model'][key], rtol=1e-5, atol=1e-6)
    assert after['optimizer']['state']
    assert after['replay_signature']['value_fraction'] == value_fraction
    if sampling == 'uniform':
        chained = tmp_path / 'chained'; run(chained, resumed / 'final.pt')
        chain = torch.load(chained / 'final.pt', map_location='cpu', weights_only=False)
        assert chain['base_total_updates'] == 2 and chain['total_updates'] == 4
        info = json.loads((chained / 'info.json').read_text())
        assert info['optimizer_carried_from_input']
        assert chain['replay_signature']['latest_fraction'] is None
