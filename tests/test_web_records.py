import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alphago.models import GoNet, NetConfig
from alphago.sgf import parse_games
from alphago.web_records import save_finished_game, search_record

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from play_server import Game


@pytest.fixture
def game(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    g = Game(GoNet(NetConfig(blocks=1, channels=4)).eval(), save_dir=tmp_path, playouts=0)
    g.model_name = 'test model'
    monkeypatch.setattr(g, 'judge', lambda: {'score': -7.5, 'dead': []})
    yield g
    g.close()


def root():
    return SimpleNamespace(moves=np.array([0, 361]), N=np.array([1., 3.]), W=np.array([.2, .6]))


def test_terminal_game_saved_once_with_real_search_and_provenance(game, tmp_path):
    r = root()
    game.player = SimpleNamespace(genmove=lambda board: 361, last_root=r)
    game.engine_move()
    assert not list(tmp_path.iterdir())
    game.engine_move()
    assert game.saved_game and game.save_error is None
    game.autosave()
    assert len(list(tmp_path.iterdir())) == 3
    path = next(tmp_path.glob('*.npz'))
    with np.load(path) as d:
        assert d['moves'].tolist() == [361, 361]
        assert d['train_mask'].tolist() == [True, True]
        assert d['visits'].tolist() == [4, 4]
        np.testing.assert_allclose(d['search_values'], [.2, .2])
        np.testing.assert_allclose(d['policy'].sum(axis=1), [1., 1.])
        assert d['features'].shape == (2, 34, 19, 19)
        assert bool(d['judged']) and float(d['score']) == -7.5
    sgf = next(tmp_path.glob('*.sgf')).read_text()
    assert 'RE[W+7.5]' in sgf and 'visits=4' in sgf
    metadata = json.loads(next(tmp_path.glob('*.json')).read_text())
    assert metadata['training_positions'] == 2


def test_human_moves_never_become_policy_targets_even_with_analysis(game, tmp_path):
    game._analysis_record = search_record(root())
    game.play(361)
    game.play(361)
    with np.load(next(tmp_path.glob('*.npz'))) as d:
        assert not d['train_mask'].any()
        assert d['visits'].tolist() == [4, 0]
        assert np.isnan(d['search_values'][1])


def test_undo_trims_provenance_and_keeps_archived_branch(game, tmp_path):
    game.play(361)
    game.play(361)
    first = game.saved_game
    game.history.pop(); game.rebuild()
    assert len(game.move_records) == 1 and game.saved_game is None
    game.play(0); game.play(361); game.play(361)
    assert game.saved_game != first
    assert len(list(tmp_path.glob('*.sgf'))) == 2


def test_failed_referee_does_not_invent_a_training_label(game, tmp_path, monkeypatch):
    monkeypatch.setattr(game, 'judge', lambda: None)
    game.play(361); game.play(361)
    with np.load(next(tmp_path.glob('*.npz'))) as d:
        assert not bool(d['judged']) and np.isnan(d['score'])
    assert 'RE[Void]' in next(tmp_path.glob('*.sgf')).read_text()


def test_sgf_escaping_and_incomplete_game_rejection(tmp_path):
    with pytest.raises(ValueError, match='completed'):
        save_finished_game(tmp_path, 'a' * 32, [0], [{'source': 'human'}], 7.5, None, 'test')
    path = save_finished_game(tmp_path, 'a' * 32, [361, 361], [{'source': 'imported'}] * 2,
                              6.5, {'score': 0, 'dead': []}, r'label]with\slashes')
    text = Path(path).read_text()
    assert 'KM[6.5]' in text and 'RE[0]' in text
    assert r'label\]with\\slashes' in text
    parsed = parse_games(text)[0]
    assert parsed.komi == 6.5 and len(parsed.moves) == 2


def test_disk_error_is_visible_without_rejecting_final_move(game, monkeypatch):
    import play_server
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(play_server, 'save_finished_game', fail)
    game.play(361); game.play(361)
    assert game.board.game_over
    assert game.saved_game is None and game.save_error == 'disk full'
