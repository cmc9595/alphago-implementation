# go/zobrist.py
from __future__ import annotations
import random
from typing import Dict, Tuple

from go.constants import Stone

_ZOBRIST_CACHE: Dict[Tuple[int, int], "Zobrist"] = {}


class Zobrist:
    """
    Zobrist hashing for Go board.
    Cached by (size, seed).
    """

    def __init__(self, size: int, seed: int = 0):
        self.size = size
        self.seed = seed

        rng = random.Random(seed)

        # table[y][x][stone]
        self.table = [
            [
                [rng.getrandbits(64) for _ in range(3)]
                for _ in range(size)
            ]
            for _ in range(size)
        ]

        self.black_to_play = rng.getrandbits(64)

    # ---------- hashing helpers ----------

    def hash_empty(self) -> int:
        return 0

    def hash_stone(self, x: int, y: int, stone: Stone) -> int:
        if stone == Stone.EMPTY:
            return 0
        return self.table[y][x][stone.value]


def get_zobrist(size: int, seed: int = 0) -> Zobrist:
    key = (size, seed)
    z = _ZOBRIST_CACHE.get(key)
    if z is not None:
        return z

    z = Zobrist(size=size, seed=seed)
    _ZOBRIST_CACHE[key] = z
    return z
