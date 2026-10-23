"""Sampling training batches from a prepared game file (see scripts/prepare_sgf.py)."""

import queue
import threading

import numpy as np

from alphago import replay_features


def symmetry_tables(n: int) -> np.ndarray:
    """perm[s][a] = where action a lands under symmetry s (s in 0..7); pass maps to pass."""
    idx = np.arange(n * n).reshape(n, n)
    perm = np.empty((8, n * n + 1), np.int64)
    for s in range(8):
        g = transform(idx, s)
        perm[s, : n * n] = np.argsort(g.ravel())
        perm[s, n * n] = n * n
    return perm


def transform(x: np.ndarray, s: int) -> np.ndarray:
    """Apply symmetry s to the last two (board) axes."""
    x = np.rot90(x, s % 4, axes=(-2, -1))
    return np.flip(x, axis=-1) if s >= 4 else x


class GameData:
    def __init__(self, paths, val_fraction=0.02, seed=0, value_komi=(5.5, 8.0)):
        """paths: one prepared .npz or a list of them (concatenated).

        Outcomes only make sense as value targets under normal komi, so results of games whose
        komi is outside `value_komi` (no-komi games, Fox's KM[0], missing) are masked to 0.
        """
        if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
            paths = [paths]
        moves, offsets, results, base = [], [np.zeros(1, np.int64)], [], 0
        for p in paths:
            d = np.load(p)
            moves.append(d["moves"])
            offsets.append(d["offsets"][1:] + base)
            komi = d["komi"]
            ok = (komi >= value_komi[0]) & (komi <= value_komi[1])  # NaN compares False
            results.append(np.where(ok, d["results"], 0).astype(np.int8))
            base += len(d["moves"])
        self.moves = np.concatenate(moves)
        self.offsets = np.concatenate(offsets)
        self.results = np.concatenate(results)
        self.size = 19
        lengths = np.diff(self.offsets)
        rng = np.random.default_rng(seed)
        is_val = rng.random(len(lengths)) < val_fraction
        self.splits = {}
        for name, mask in (("train", ~is_val), ("val", is_val)):
            games = np.flatnonzero(mask)
            self.splits[name] = (games, np.cumsum(lengths[games]))
        self.perm = symmetry_tables(self.size)

    def num_positions(self, split="train") -> int:
        return int(self.splits[split][1][-1])

    def sample(self, batch, rng, split="train", augment=True, threads=0):
        """Uniform over positions. Returns (features uint8 (B,C,N,N), move targets, outcome z for side to move)."""
        games, cum = self.splits[split]
        r = rng.integers(0, cum[-1], batch)
        gi = np.searchsorted(cum, r, side="right")
        ply = r - np.concatenate([[0], cum])[gi]
        game = games[gi]
        starts = self.offsets[game]
        feats = replay_features(self.moves, starts, ply.astype(np.int32), self.size, threads)
        targets = self.moves[starts + ply].astype(np.int64)
        z = self.results[game].astype(np.float32) * np.where(ply % 2 == 0, 1.0, -1.0).astype(np.float32)
        if augment:
            syms = rng.integers(0, 8, batch)
            for s in range(1, 8):
                sel = np.flatnonzero(syms == s)
                if len(sel):
                    feats[sel] = transform(feats[sel], s)
                    targets[sel] = self.perm[s][targets[sel]]
        return feats, targets, z

    def iterate(self, batch, seed, split="train", augment=True, prefetch=4):
        """Endless batch stream produced on a background thread (the C++ replay releases the GIL)."""
        q: queue.Queue = queue.Queue(maxsize=prefetch)
        stop = threading.Event()

        def worker():
            rng = np.random.default_rng(seed)
            while not stop.is_set():
                q.put(self.sample(batch, rng, split, augment))

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        try:
            while True:
                yield q.get()
        finally:
            stop.set()
