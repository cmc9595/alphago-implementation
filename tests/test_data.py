import random

import numpy as np

from alphago import Board, replay_features
from alphago.data import GameData, symmetry_tables, transform
from alphago.sgf import parse_games

SGF = """(;GM[1]FF[4]SZ[19]KM[6.5]RE[W+R]PB[a]BR[7d]PW[b]WR[8d]C[comment with \\] and (parens)]
;B[pd];W[dp]C[x](;B[pp];W[dd];B[]  ;W[tt])(;B[qq];W[aa]))
(;SZ[19]HA[2]AB[pd][dp];W[pp])"""


def test_parse_main_line_and_collection():
    g, h = parse_games(SGF)
    assert g.komi == 6.5 and g.result == -1 and g.black_rank == "7d"
    # pd -> x=15, row 3 from top -> y=15 ; dp -> x=3, y=3
    assert g.moves[:4] == [(1, 15 * 19 + 15), (2, 3 * 19 + 3), (1, 3 * 19 + 15), (2, 15 * 19 + 3)]
    assert g.moves[4:] == [(1, 361), (2, 361)]  # empty and 'tt' are both pass
    assert h.handicap == 2 and len(h.setup_black) == 2


def random_game(rng, length):
    b, moves = Board(19), []
    for _ in range(length):
        legal = np.flatnonzero(b.sensible_mask()[:-1])
        a = int(rng.choice(list(legal))) if len(legal) else 361
        b.play(a)
        moves.append(a)
    return moves


def test_replay_features_matches_board():
    rng = random.Random(0)
    moves = np.array(random_game(rng, 150), np.int16)
    plies = np.array([0, 1, 37, 149], np.int32)
    feats = replay_features(moves, np.zeros(4, np.int64), plies)
    for i, k in enumerate(plies):
        b = Board(19)
        for a in moves[:k]:
            b.play(int(a))
        np.testing.assert_array_equal(feats[i], b.features())


def test_features_are_symmetry_equivariant():
    perm = symmetry_tables(19)
    rng = random.Random(1)
    for _ in range(3):
        moves = random_game(rng, 120)
        base = replay_features(np.array(moves, np.int16), np.zeros(1, np.int64), np.array([120], np.int32))[0]
        for s in range(8):
            mapped = np.array([perm[s][a] for a in moves], np.int16)
            got = replay_features(mapped, np.zeros(1, np.int64), np.array([120], np.int32))[0]
            np.testing.assert_array_equal(got, transform(base, s))


def test_game_data_sampling(tmp_path):
    rng = random.Random(2)
    games = [random_game(rng, rng.randint(60, 120)) for _ in range(20)]
    moves = np.concatenate([np.array(g, np.int16) for g in games])
    offsets = np.concatenate([[0], np.cumsum([len(g) for g in games])]).astype(np.int64)
    results = np.array([1, -1] * 10, np.int8)
    path = tmp_path / "g.npz"
    np.savez(path, moves=moves, offsets=offsets, results=results, komi=np.full(20, 7.5, np.float32))

    data = GameData(path, val_fraction=0.2)
    assert data.num_positions("train") + data.num_positions("val") == len(moves)
    feats, targets, z = data.sample(64, np.random.default_rng(0), augment=True)
    assert feats.shape == (64, 34, 19, 19)
    # Every target must be a legal (sensible-plane) point in its own, augmented features.
    for f, t in zip(feats, targets):
        if t != 361:
            assert f[32].ravel()[t] == 1
    assert set(np.unique(z)) <= {-1.0, 1.0}
