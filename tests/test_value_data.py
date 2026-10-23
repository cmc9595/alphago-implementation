import numpy as np
import torch

from alphago import Board
from alphago.gnugo import ownership
from alphago.models import GoNet, NetConfig
from alphago.value_data import Source, ValueData


def random_game(rng, length):
    b, moves = Board(19), []
    for _ in range(length):
        legal = np.flatnonzero(b.sensible_mask()[:-1])
        a = int(rng.choice(legal)) if len(legal) else 361
        b.play(a)
        moves.append(a)
    return moves


def make_source(rng, games=6):
    gs = [random_game(rng, int(rng.integers(40, 80))) for _ in range(games)]
    moves = np.concatenate([np.array(g, np.int16) for g in gs])
    offsets = np.concatenate([[0], np.cumsum([len(g) for g in gs])]).astype(np.int64)
    own = np.stack([ownership(g, []) for g in gs])
    z = np.array([1, -1, 1, -1, 0, 1][:games], np.float32)
    plies = [np.array([5, 20, len(g) - 1]) for g in gs]
    return Source("t", moves, offsets, z, own, plies)


def test_targets_are_side_to_move_and_augmentation_is_consistent():
    rng = np.random.default_rng(0)
    data = ValueData([make_source(rng)], test_fraction=0.0)
    rows = data.index["train"]
    feats, z, own = data.batch_from_index(rows)
    src = data.sources[0]
    for (si, g, p), zz in zip(rows, z):
        assert zz == src.z_black[g] * (1 if p % 2 == 0 else -1)
    # Ownership must rotate with the board: under augmentation, the "own stones" plane (0) and the
    # ownership map are transformed by the same symmetry, so their agreement is unchanged.
    feats_a, _, own_a = data.batch_from_index(rows, np.random.default_rng(1), augment=True)
    agree = lambda f, o: ((f[:, 0] == 1) & (o > 0)).sum()  # own stones on own territory
    assert agree(feats, own) == agree(feats_a, own_a)


def test_ownership_head_shapes():
    net = GoNet(NetConfig(blocks=1, channels=8, policy=False, ownership=True)).eval()
    x = torch.zeros(3, 34, 19, 19)
    p, v, o = net.forward_all(x)
    assert p is None and v.shape == (3,) and o.shape == (3, 19, 19)
    assert net(x)[1].shape == (3,)  # MCTS keeps using forward() -> (policy, value)
