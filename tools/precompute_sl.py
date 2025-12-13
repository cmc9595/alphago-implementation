# tools/precompute_sl.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from data.sgf_dataset import iter_sl_samples


@dataclass
class PrecomputeConfig:
    zip_dir: Path = Path("data")
    out_dir: Path = Path("precomputed/sl_kgs_alpha48")
    board_size: int = 19
    include_pass: bool = False
    max_games: int | None = None

    shard_size: int = 4096            # samples per shard
    compress_dtype: str = "uint8"     # "uint8" or "float16" or "float32"


def _pack_x(x: np.ndarray, dtype: str) -> torch.Tensor:
    # x: (48,19,19) float32 0/1 mostly (some planes can be small ints)
    if dtype == "uint8":
        # store 0..255
        return torch.from_numpy((x * 255.0).clip(0, 255).astype(np.uint8))
    if dtype == "float16":
        return torch.from_numpy(x.astype(np.float16))
    return torch.from_numpy(x.astype(np.float32))


def main(cfg: PrecomputeConfig):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    xs: List[torch.Tensor] = []
    ys: List[int] = []
    shard_idx = 0
    n_samples = 0

    it = iter_sl_samples(
        zip_dir=cfg.zip_dir,
        board_size=cfg.board_size,
        include_pass=cfg.include_pass,
        max_games=cfg.max_games,
    )

    for sample in tqdm(it, desc="precompute"):
        xs.append(_pack_x(sample.x, cfg.compress_dtype))
        ys.append(int(sample.y))
        n_samples += 1

        if len(xs) >= cfg.shard_size:
            x_t = torch.stack(xs, dim=0)  # (N,48,19,19)
            y_t = torch.tensor(ys, dtype=torch.long)
            out_path = cfg.out_dir / f"shard_{shard_idx:05d}.pt"
            torch.save({"x": x_t, "y": y_t}, out_path)
            xs.clear()
            ys.clear()
            shard_idx += 1

    # flush tail
    if xs:
        x_t = torch.stack(xs, dim=0)
        y_t = torch.tensor(ys, dtype=torch.long)
        out_path = cfg.out_dir / f"shard_{shard_idx:05d}.pt"
        torch.save({"x": x_t, "y": y_t}, out_path)

    # manifest
    manifest = {
        "zip_dir": str(cfg.zip_dir),
        "board_size": cfg.board_size,
        "include_pass": cfg.include_pass,
        "compress_dtype": cfg.compress_dtype,
        "shard_size": cfg.shard_size,
        "num_samples": n_samples,
    }
    torch.save(manifest, cfg.out_dir / "manifest.pt")
    print(f"[done] samples={n_samples} shards~{shard_idx+1} out={cfg.out_dir}")


if __name__ == "__main__":
    cfg = PrecomputeConfig()
    main(cfg)
