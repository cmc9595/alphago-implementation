# train/sl_dataset.py
from __future__ import annotations
import torch
from torch.utils.data import IterableDataset
from pathlib import Path

from data.sgf_dataset import iter_sl_samples


class SLIterableDataset(IterableDataset):
    def __init__(
        self,
        zip_dir: Path,
        board_size: int = 19,
        skip_handicap: bool = True,
    ):
        super().__init__()
        self.zip_dir = zip_dir
        self.board_size = board_size
        self.skip_handicap = skip_handicap

    def __iter__(self):
        for sample in iter_sl_samples(
            zip_dir=self.zip_dir,
            board_size=self.board_size,
            skip_handicap=self.skip_handicap,
        ):
            x = torch.from_numpy(sample.x)      # (48,19,19)
            y = torch.tensor(sample.y).long()   # scalar
            yield x, y
