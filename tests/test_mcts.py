import numpy as np

from alphago import Board, Color
from alphago.mcts import MCTS


def score_evaluator(boards):
    """Uniform priors; value = tanh(area score / 10) from the side to move's view."""
    n = boards[0].size
    logits = np.zeros((len(boards), n * n + 1), np.float32)
    values = []
    for b in boards:
        s = b.score()
        values.append(np.tanh((s if b.to_move == Color.BLACK else -s) / 10))
    return logits, np.array(values, np.float32)


def test_visit_counts_add_up():
    m = MCTS(score_evaluator, playouts=64, batch=8)
    root = m.search(Board(9))
    assert root.N.sum() == 64
    assert np.all(root.N >= 0)


def test_finds_capture_of_big_group():
    # White's 4-stone group on C3..F3 has a single liberty at G3; Black to move should take it.
    b = Board(9, komi=0.5)
    for c in [(2, 2), (3, 2), (4, 2), (5, 2)]:
        b.place_stone(c[1] * 9 + c[0], Color.WHITE)
    for c in [(1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (2, 3), (3, 3), (4, 3), (5, 3)]:
        b.place_stone(c[1] * 9 + c[0], Color.BLACK)
    capture = 2 * 9 + 6  # G3
    m = MCTS(score_evaluator, playouts=400, batch=8)
    assert m.genmove(b) == capture


def test_two_pass_leaf_stops_search_and_reuses_terminal_value():
    # Force pass. Even with many visits the search must never play a third move
    # after the two passes, nor overwrite the terminal value with later states.
    seen = []
    def evaluator(boards):
        logits = np.full((len(boards), 82), -100., np.float32)
        logits[:, 81] = 100.
        values = []
        for board in boards:
            seen.append((board.move_number, board.consecutive_passes))
            assert board.move_number <= 2
            values.append(.75 if board.game_over else 0.)
        return logits, np.asarray(values, np.float32)
    b = Board(9)
    b.play(81)
    root = MCTS(evaluator, playouts=64, batch=8).search(b)
    i = int(np.flatnonzero(root.moves == 81)[0])
    assert root.N.sum() == 64
    assert root.N[i] == 64
    assert np.isclose(root.W[i] / root.N[i], -.75)
    leaf = root.children[i]
    assert leaf.moves.size == 0 and not leaf.children
    # Root plus at most one batch of duplicate terminal evaluations.
    assert len(seen) <= 9


def test_terminal_backup_uses_correct_colour_on_both_turns():
    for first in (Color.BLACK, Color.WHITE):
        b = Board(9, komi=7.5)
        b.set_to_move(first)
        b.play(81)
        def evaluator(boards):
            logits = np.full((len(boards), 82), -100., np.float32)
            logits[:, 81] = 100.
            # Empty board: White wins. Values are from each leaf's player view.
            values = np.array([1. if x.to_move == Color.WHITE else -1. for x in boards], np.float32)
            return logits, values
        root = MCTS(evaluator, playouts=32, batch=4).search(b)
        i = int(np.flatnonzero(root.moves == 81)[0])
        expected = 1. if b.to_move == Color.WHITE else -1.
        assert np.isclose(root.W[i] / root.N[i], expected)


def test_finished_root_is_rejected_without_network_evaluation():
    import pytest
    b = Board(9)
    b.play(81); b.play(81)
    def evaluator(boards):
        raise AssertionError('finished root must not be evaluated')
    with pytest.raises(ValueError, match='finished game'):
        MCTS(evaluator).search(b)
