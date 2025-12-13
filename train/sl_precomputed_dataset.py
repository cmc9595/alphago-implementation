# train/sl_precomputed_dataset.py
from __future__ import annotations

from pathlib import Path
import random
import torch
from torch.utils.data import IterableDataset


class SLPrecomputedIterableDataset(IterableDataset):
    def __init__(self, precomp_dir: Path, shuffle_shards: bool = True, shuffle_in_shard: bool = True, seed: int = 0):
        super().__init__()
        self.precomp_dir = precomp_dir
        self.shuffle_shards = shuffle_shards
        self.shuffle_in_shard = shuffle_in_shard
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        shards = sorted(self.precomp_dir.glob("shard_*.pt"))
        if self.shuffle_shards:
            rng.shuffle(shards)

        for sp in shards:
            data = torch.load(sp, map_location="cpu")
            x = data["x"]  # (N,48,19,19) uint8/float16/float32
            y = data["y"]  # (N,)

            idx = list(range(x.shape[0]))
            if self.shuffle_in_shard:
                rng.shuffle(idx)

            for i in idx:
                xi = x[i]
                # decode uint8 back to float32 0..1
                if xi.dtype == torch.uint8:
                    xi = xi.float().div_(255.0)
                else:
                    xi = xi.float()
                yield xi, y[i]
