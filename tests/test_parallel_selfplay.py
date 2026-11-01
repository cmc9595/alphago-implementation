"""Parallel self-play driver: correctness with a fake (uniform) evaluator, no GPU needed."""
import numpy as np

from alphago.parallel_selfplay import ParallelConfig, ParallelSelfPlay
from alphago.search_learning import SearchPlayConfig


def fake_eval(boards):
    n = len(boards)
    return np.zeros((n, boards[0].size ** 2 + 1), np.float32), np.zeros(n, np.float32)


def run(**kw):
    play = SearchPlayConfig(visits=24, fast_visits=8, full_search_probability=.5, board_size=9, max_moves=40)
    cfg = ParallelConfig(play=play, parallel_games=3, leaves_per_game=4, **kw)
    finished = []
    ParallelSelfPlay(fake_eval, cfg, [(i, 100 + i) for i in range(5)], finished.append).run()
    return finished


def test_all_games_finish_with_exact_visit_counts():
    games = run()
    assert sorted(g.gid for g in games) == list(range(5))
    for g in games:
        rec = g.record()
        assert len(rec['moves']) == len(rec['policy']) == len(rec['features']) == len(rec['train_mask'])
        assert np.allclose(rec['policy'].sum(1), 1)
        expected = np.where(rec['train_mask'], 24, 8)
        assert (rec['visits'] == expected).all()


def test_reuse_tree_keeps_full_search_targets_fresh():
    for g in run(reuse_tree=True):
        rec = g.record()
        assert (rec['visits'][rec['train_mask']] == 24).all()
        assert (rec['visits'][~rec['train_mask']] >= 8).all()


def test_same_seed_is_deterministic():
    a = [g.record()['moves'].tolist() for g in sorted(run(), key=lambda g: g.gid)]
    b = [g.record()['moves'].tolist() for g in sorted(run(), key=lambda g: g.gid)]
    assert a == b


def test_fast_opening_sample_diversifies_openings():
    def openings(flag):
        play = SearchPlayConfig(visits=24, fast_visits=120, full_search_probability=.0001, board_size=9, max_moves=12, temperature_moves=10)
        cfg = ParallelConfig(play=play, parallel_games=4, leaves_per_game=4, fast_opening_sample=flag)
        out = []
        ParallelSelfPlay(fake_eval, cfg, [(i, 500 + i) for i in range(8)], out.append).run()
        return {tuple(g.record()['moves'][:10].tolist()) for g in out}
    assert len(openings(False)) == 1   # uniform priors + greedy fast moves: every game identical
    assert len(openings(True)) > 1
