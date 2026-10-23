import numpy as np
import torch

from alphago import Board
from alphago.models import GoNet, NetConfig
from alphago.selfplay import SelfPlayConfig, cleanup_mask, play_batch


def tiny_net(seed):
    torch.manual_seed(seed)
    return GoNet(NetConfig(board_size=9, blocks=1, channels=8)).eval()


def test_play_batch_finishes_and_replays():
    cfg = SelfPlayConfig(board_size=9, max_moves=120, cleanup_max=80)
    recs = play_batch(tiny_net(0), tiny_net(1), 6, np.random.default_rng(0), torch.device("cpu"), cfg)
    assert len(recs) == 6
    assert [r.learner_black for r in recs] == [True, False] * 3
    for r in recs:
        assert 0 < len(r.moves) <= cfg.max_moves
        assert len(r.cleanup) <= cfg.cleanup_max
        # Replaying normal + cleanup moves must be legal and reproduce the final score.
        b = Board(9, cfg.komi)
        for a in r.moves:
            assert b.is_legal(a)
            b.play(a)
        assert abs(b.score() - r.score_before) < 1e-6
        for a in r.cleanup:
            assert b.is_legal(a)
            b.play(a)
        assert abs(b.score() - r.score_after) < 1e-6
        # Cleanup only plays in contested regions, and passes only when no such move is left.
        b2 = Board(9, cfg.komi)
        for a in r.moves:
            b2.play(a)
        for a in r.cleanup:
            allowed = cleanup_mask(b2)
            assert allowed[a]
            if a == 81:
                assert not allowed[:81].any()
            b2.play(a)


def test_learner_won_uses_cleanup_result():
    cfg = SelfPlayConfig(board_size=9, max_moves=60, cleanup_max=40)
    recs = play_batch(tiny_net(2), tiny_net(3), 4, np.random.default_rng(1), torch.device("cpu"), cfg)
    for r in recs:
        assert r.learner_won == ((r.score_after > 0) == r.learner_black)


def test_cleanup_never_fills_own_territory():
    from alphago import Color
    # Black wall on row 3 owns rows 1-2; nothing contested there, so cleanup must pass.
    b = Board(9)
    for x in range(9):
        b.place_stone(2 * 9 + x, Color.BLACK)
        b.place_stone(4 * 9 + x, Color.WHITE)
    m = cleanup_mask(b)
    assert not m[:18].any()  # Black's own area (rows 1-2) is off limits
    assert m[3 * 9 + 4]  # the contested row between the walls is allowed
