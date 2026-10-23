"""Training positions for the value network (+ optional ownership head).

Three sources, each a set of games with a final result and GNU Go ownership map:
  gen    data/value/gen/batch_*.npz         AlphaGo-style generated games, labelled plies after U
  rl     checkpoints/rl_6x96/games/*.npz    RL self-play archive, a few random plies per game
  human  data/value/human_kgs.npz           finished KGS games confirmed by the recorded score

A sample is (game, ply): the position after `ply` moves. Targets are from the side to move:
value z = +1 if it wins under komi 7.5, ownership = final ownership flipped to its perspective.
Whole games are held out for testing so no test position shares a game with training.
"""

import glob
import queue
import threading
from dataclasses import dataclass

import numpy as np

from alphago import replay_features
from alphago.data import symmetry_tables, transform

KOMI = 7.5


@dataclass
class Source:
    name: str
    moves: np.ndarray  # int16, all games concatenated
    offsets: np.ndarray  # int64, len = games + 1
    z_black: np.ndarray  # float32 per game: +1 black wins under komi 7.5, -1 white, 0 unusable
    ownership: np.ndarray  # int8 (games, 361), +1 black / -1 white / 0 neutral
    plies: list  # per game: array of labelled plies
    score_black: np.ndarray = None  # float32 per game: final margin for Black under komi 7.5 (NaN = unknown)


def _concat(files, keys):
    parts = {k: [] for k in keys}
    offsets, base = [np.zeros(1, np.int64)], 0
    for f in files:
        d = np.load(f)
        for k in keys:
            parts[k].append(d[k])
        offsets.append(d["offsets"][1:] + base)
        base += len(d["moves"])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["offsets"] = np.concatenate(offsets)
    return out


def load_gen(pattern="data/value/gen/batch_*.npz", name="gen"):
    files = sorted(glob.glob(pattern))
    d = _concat(files, ["moves", "plies", "score", "ownership"])
    plies = [p[p >= 0] for p in d["plies"]]
    z = np.sign(d["score"]).astype(np.float32)
    return Source(name, d["moves"], d["offsets"], z, d["ownership"], plies, d["score"].astype(np.float32))


def load_rl(pattern="checkpoints/rl_6x96/games/iter*.npz", per_game=2, seed=0):
    files = sorted(glob.glob(pattern))
    d = _concat(files, ["moves", "score", "judged", "ownership"])
    rng = np.random.default_rng(seed)
    lengths = np.diff(d["offsets"])
    plies = [np.sort(rng.choice(L, min(per_game, L), replace=False)) for L in lengths]
    z = np.where(d["judged"], np.sign(d["score"]), 0).astype(np.float32)
    return Source("rl", d["moves"], d["offsets"], z, d["ownership"], plies)


def load_human(path="data/value/human_kgs.npz", per_game=8, seed=0, margin=1.0):
    d = np.load(path)
    rng = np.random.default_rng(seed)
    lengths = np.diff(d["offsets"])
    plies = [np.sort(rng.choice(L, min(per_game, L), replace=False)) for L in lengths]
    # Area-scoring ownership does not depend on komi, so the recorded result converts to komi 7.5.
    # Results within `margin` of the line are too close to call across counting rules: no value label.
    adjusted = d["score"] + d["komi"] - KOMI
    z = np.where(np.abs(adjusted) > margin, np.sign(adjusted), 0).astype(np.float32)
    return Source("human", d["moves"], d["offsets"], z, d["ownership"], plies)


class ValueData:
    def __init__(self, sources, test_fraction=0.03, seed=0, max_score_gap=None):
        self.sources = sources
        self.perm = symmetry_tables(19)
        rng = np.random.default_rng(seed)
        self.index = {"train": [], "test": []}  # (source id, game, ply)
        for si, src in enumerate(sources):
            games = len(src.offsets) - 1
            is_test = rng.random(games) < test_fraction
            for g in range(games):
                for p in src.plies[g]:
                    self.index["test" if is_test[g] else "train"].append((si, g, int(p)))
        for k in self.index:
            self.index[k] = np.array(self.index[k], np.int64).reshape(-1, 3)

    def size(self, split="train"):
        return len(self.index[split])

    def batch_from_index(self, rows, rng=None, augment=False):
        n = len(rows)
        feats = np.empty((n, 34, 19, 19), np.uint8)
        z = np.empty(n, np.float32)
        own = np.empty((n, 19, 19), np.float32)
        for si in np.unique(rows[:, 0]):
            sel = np.flatnonzero(rows[:, 0] == si)
            src = self.sources[si]
            g, p = rows[sel, 1], rows[sel, 2]
            starts = src.offsets[g]
            feats[sel] = replay_features(src.moves, starts, p.astype(np.int32))
            to_move = np.where(p % 2 == 0, 1.0, -1.0).astype(np.float32)  # +1 Black to move
            z[sel] = src.z_black[g] * to_move
            own[sel] = src.ownership[g].reshape(-1, 19, 19) * to_move[:, None, None]
        if augment:
            syms = rng.integers(0, 8, n)
            for s in range(1, 8):
                sel = np.flatnonzero(syms == s)
                if len(sel):
                    feats[sel] = transform(feats[sel], s)
                    own[sel] = transform(own[sel], s)
        return feats, z, own

    def scores_from_index(self, rows):
        """Final score margin from the side to move's view (NaN where the source has no score)."""
        out = np.full(len(rows), np.nan, np.float32)
        for si in np.unique(rows[:, 0]):
            src = self.sources[si]
            if src.score_black is None:
                continue
            sel = np.flatnonzero(rows[:, 0] == si)
            to_move = np.where(rows[sel, 2] % 2 == 0, 1.0, -1.0)
            out[sel] = src.score_black[rows[sel, 1]] * to_move
        return out

    def sample(self, batch, rng, split="train", augment=True):
        rows = self.index[split][rng.integers(0, self.size(split), batch)]
        return self.batch_from_index(rows, rng, augment)

    def iterate(self, batch, seed, prefetch=4):
        q: queue.Queue = queue.Queue(maxsize=prefetch)

        def worker():
            rng = np.random.default_rng(seed)
            while True:
                q.put(self.sample(batch, rng))

        threading.Thread(target=worker, daemon=True).start()
        while True:
            yield q.get()
