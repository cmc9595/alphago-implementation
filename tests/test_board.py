import random

import numpy as np
import pytest

from alphago import NUM_FEATURES, Board, Color, features_batch

COLS = "ABCDEFGHJKLMNOPQRST"

# Plane indices, mirroring engine/src/board.hpp.
OWN, OPP, EMPTY, ONES = 0, 1, 2, 3
OWN_LIBS, OPP_LIBS = 12, 16
CAPTURE_SIZE, SELF_ATARI_SIZE, LIBS_AFTER = 20, 23, 26
LADDER_CAPTURE, LADDER_ESCAPE, SENSIBLE, BLACK_TO_MOVE = 30, 31, 32, 33


def a(coord, n=19):
    """GTP coordinate like 'D4' -> action index."""
    x = COLS.index(coord[0].upper())
    y = int(coord[1:]) - 1
    return y * n + x


def setup(n, black=(), white=(), to_move=Color.BLACK):
    b = Board(n)
    for c in black:
        b.place_stone(a(c, n), Color.BLACK)
    for c in white:
        b.place_stone(a(c, n), Color.WHITE)
    b.set_to_move(to_move)
    return b


def test_capture_and_counts():
    b = setup(9, black=["D5", "F5", "E6"], white=["E5"])
    b.play(a("E4", 9))
    assert b.color_at(a("E5", 9)) == Color.EMPTY
    assert b.captures(Color.BLACK) == 1


def test_suicide_is_illegal():
    b = setup(9, black=["A2", "B1"], to_move=Color.WHITE)
    assert not b.is_legal(a("A1", 9))


def test_capture_beats_suicide():
    # White at A1 has no liberties itself but captures the black stone at B1.
    b = setup(9, black=["B1"], white=["C1", "B2", "A2"])
    b.set_to_move(Color.WHITE)
    assert b.is_legal(a("A1", 9))


def test_simple_ko():
    # Textbook ko: black E5 captures the white stone at D5.
    b = setup(9, black=["C5", "D6", "D4"], white=["E6", "E4", "F5", "D5"])
    b.play(a("E5", 9))  # black captures D5
    assert b.color_at(a("D5", 9)) == Color.EMPTY
    assert b.ko_action == a("D5", 9)
    assert not b.is_legal(a("D5", 9))  # white may not retake immediately
    b.play(a("J9", 9))
    b.play(a("J8", 9))
    assert b.is_legal(a("D5", 9))


def ko_after_passes(superko):
    # Black takes the ko, both pass (clearing the simple-ko ban), then White
    # retaking would recreate the position from before Black's capture.
    b = Board(9, superko=superko)
    for c in ["C5", "D6", "D4"]:
        b.place_stone(a(c, 9), Color.BLACK)
    for c in ["E6", "E4", "F5", "D5"]:
        b.place_stone(a(c, 9), Color.WHITE)
    b.set_to_move(Color.BLACK)
    b.play(a("E5", 9))
    b.play(b.pass_action)
    b.play(b.pass_action)
    return b


def test_superko_forbids_repeating_a_position():
    b = ko_after_passes(superko=True)
    assert b.ko_action == -1
    assert not b.is_legal(a("D5", 9))
    assert b.legal_mask()[a("D5", 9)] == 0
    assert b.sensible_mask()[a("D5", 9)] == 0
    with pytest.raises(ValueError):
        b.play(a("D5", 9))
    assert b.is_legal(a("J9", 9))
    assert b.legal_mask()[b.pass_action] == 1


def test_superko_off_by_default():
    b = ko_after_passes(superko=False)
    assert not b.superko
    assert b.is_legal(a("D5", 9))
    b.play(a("D5", 9))
    assert b.color_at(a("E5", 9)) == Color.EMPTY


def test_superko_history_is_per_copy():
    b = ko_after_passes(superko=True)
    c = b.copy()
    assert c.superko and not c.is_legal(a("D5", 9))
    # Positions played on the copy must not leak into the original.
    c.play(a("J9", 9))
    c.play(a("A9", 9))
    assert b.is_legal(a("J9", 9))


def test_superko_random_games_never_repeat():
    rng = random.Random(7)
    for _ in range(30):
        b = Board(5, superko=True)
        seen = {b.stones().tobytes()}
        for _ in range(150):
            mask = b.legal_mask()
            moves = np.flatnonzero(mask[:-1])
            if len(moves) == 0:
                break
            b.play(int(rng.choice(moves)))
            key = b.stones().tobytes()
            assert key not in seen
            seen.add(key)


def test_two_passes_end_game_and_score():
    b = Board(9, 7.5)
    assert b.score() == -7.5
    b.play(a("E5", 9))
    assert b.score() == 81 - 7.5  # a lone stone owns the whole empty board
    b.play(b.pass_action)
    b.play(b.pass_action)
    assert b.game_over


def test_eye_is_not_sensible():
    b = setup(9, black=["A2", "B1", "B2"])
    sens = b.sensible_mask()
    assert b.is_legal(a("A1", 9))
    assert sens[a("A1", 9)] == 0
    assert sens[a("E5", 9)] == 1


def test_feature_shape_and_basic_planes():
    b = setup(9, black=["D4"], white=["E5"])
    f = b.features()
    assert f.shape == (NUM_FEATURES, 9, 9)
    assert f[OWN].sum() == 1 and f[OPP].sum() == 1
    assert f[EMPTY].sum() == 79
    assert f[ONES].all()
    assert f[BLACK_TO_MOVE].all()
    assert f[OWN_LIBS + 3, 3, 3] == 1  # D4 has 4 liberties
    b.play(a("C3", 9))
    f = b.features()
    assert not f[BLACK_TO_MOVE].any()
    assert f[4, 2, 2] == 1  # history plane 0 marks the last move


def test_capture_and_self_atari_planes():
    b = setup(9, black=["D5", "F5", "E6"], white=["E5", "A2", "B1"])
    f = b.features()
    assert f[CAPTURE_SIZE + 0, 3, 4] == 1  # E4 captures one stone
    assert f[SELF_ATARI_SIZE + 0, 0, 0] == 0  # A1 is illegal (suicide) for black
    b2 = setup(9, black=["D1"], white=["C1", "E2", "D3", "C2"])
    f2 = b2.features()
    assert f2[SELF_ATARI_SIZE + 1, 1, 3] == 1  # D2 joins D1 into a 2-stone chain in atari


def at(f, plane, coord, n=19):
    i = a(coord, n)
    return f[plane, i // n, i % n]


def test_ladder_capture():
    # White D4 with black C4, D5, E3: atari at E4 chases white toward the empty A1 corner.
    b = setup(19, black=["C4", "D5", "E3"], white=["D4"])
    assert at(b.features(), LADDER_CAPTURE, "E4") == 1
    # A white stone on the ladder's diagonal breaks it.
    broken = setup(19, black=["C4", "D5", "E3"], white=["D4", "B2"])
    assert at(broken.features(), LADDER_CAPTURE, "E4") == 0


def test_ladder_escape():
    # White D4 already in atari; extending to D3 runs into a working ladder.
    b = setup(19, black=["C4", "D5", "E4", "E3"], white=["D4"], to_move=Color.WHITE)
    assert at(b.features(), LADDER_ESCAPE, "D3") == 0
    b2 = setup(19, black=["C4", "D5", "E4", "E3"], white=["D4", "B2"], to_move=Color.WHITE)
    assert at(b2.features(), LADDER_ESCAPE, "D3") == 1


def test_features_batch_matches_single():
    rng = random.Random(1)
    boards = []
    for _ in range(16):
        b = Board(19)
        for _ in range(rng.randint(0, 120)):
            moves = np.flatnonzero(b.sensible_mask()[:-1])
            if len(moves) == 0:
                break
            b.play(int(rng.choice(list(moves))))
        boards.append(b)
    batch = features_batch(boards, threads=4)
    for i, b in enumerate(boards):
        np.testing.assert_array_equal(batch[i], b.features())


# --- Cross-check against a slow, obviously-correct reference implementation ---


class RefBoard:
    def __init__(self, n):
        self.n = n
        self.g = [[0] * n for _ in range(n)]
        self.ko = None
        self.to_move = 1

    def nbrs(self, x, y):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= x + dx < self.n and 0 <= y + dy < self.n:
                yield x + dx, y + dy

    def group(self, x, y):
        c = self.g[y][x]
        seen, stack, libs = {(x, y)}, [(x, y)], set()
        while stack:
            p = stack.pop()
            for q in self.nbrs(*p):
                v = self.g[q[1]][q[0]]
                if v == 0:
                    libs.add(q)
                elif v == c and q not in seen:
                    seen.add(q)
                    stack.append(q)
        return seen, libs

    def try_play(self, x, y):
        if self.g[y][x] != 0 or (x, y) == self.ko:
            return None
        c, o = self.to_move, 3 - self.to_move
        g = [row[:] for row in self.g]
        self.g, saved = g, self.g
        g[y][x] = c
        captured = []
        for q in self.nbrs(x, y):
            if g[q[1]][q[0]] == o:
                stones, libs = self.group(*q)
                if not libs:
                    captured.extend(stones)
                    for sx, sy in stones:
                        g[sy][sx] = 0
        _, libs = self.group(x, y)
        self.g = saved
        if not libs:
            return None
        stones, _ = RefBoard.group_on(g, self.n, x, y)
        ko = captured[0] if len(captured) == 1 and len(stones) == 1 and len(libs) == 1 else None
        return g, ko

    @staticmethod
    def group_on(g, n, x, y):
        r = RefBoard(n)
        r.g = g
        return r.group(x, y)

    def legal(self, x, y):
        return self.try_play(x, y) is not None

    def play(self, x, y):
        self.g, self.ko = self.try_play(x, y)
        self.to_move = 3 - self.to_move

    def pass_(self):
        self.ko = None
        self.to_move = 3 - self.to_move


@pytest.mark.parametrize("n,seed", [(9, 0), (9, 1), (13, 2), (19, 3)])
def test_random_games_match_reference(n, seed):
    rng = random.Random(seed)
    for _ in range(5 if n == 19 else 15):
        b, r = Board(n), RefBoard(n)
        for _ in range(n * n * 3):
            mask = b.legal_mask()
            for y in range(n):
                for x in range(n):
                    assert bool(mask[y * n + x]) == r.legal(x, y), (str(b), x, y)
            moves = [m for m in np.flatnonzero(mask[:-1])]
            if not moves or rng.random() < 0.02:
                b.play(b.pass_action)
                r.pass_()
                continue
            m = int(rng.choice(moves))
            b.play(m)
            r.play(m % n, m // n)
            assert (b.stones() == np.array(r.g)).all()
