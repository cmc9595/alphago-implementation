# train/train_sl.py
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from bots.nn.policy_net import PolicyNet
from train.sl_precomputed_dataset import SLPrecomputedIterableDataset


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def save_checkpoint(path: Path, model: nn.Module, optim: torch.optim.Optimizer, step: int):
    torch.save(
        {"model": model.state_dict(), "optim": optim.state_dict(), "step": int(step)},
        path
    )


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, y: torch.Tensor, ks=(1, 5, 10)) -> dict[int, float]:
    """
    logits: (B, 361 or 362)
    y: (B,)
    """
    max_k = max(ks)
    # topk idx: (B, max_k)
    topk = torch.topk(logits, k=max_k, dim=1).indices
    out: dict[int, float] = {}
    for k in ks:
        hit = (topk[:, :k] == y.unsqueeze(1)).any(dim=1).float().mean().item()
        out[k] = hit
    return out


@torch.no_grad()
def evaluate_fixed_batches(
    model: nn.Module,
    fixed_batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: str,
    ks=(1, 5, 10),
) -> dict[str, float]:
    model.eval()
    n = 0
    acc_sum = {k: 0.0 for k in ks}
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x_cpu, y_cpu in fixed_batches:
        x = x_cpu.to(device)
        y = y_cpu.to(device)

        logits = model(x)
        loss = criterion(logits, y).item()
        acc = topk_accuracy(logits, y, ks=ks)

        bs = x.shape[0]
        n += bs
        loss_sum += loss * bs
        for k in ks:
            acc_sum[k] += acc[k] * bs

    model.train()
    return {
        "val_loss": loss_sum / max(n, 1),
        **{f"val_top{k}": acc_sum[k] / max(n, 1) for k in ks},
    }


def build_fixed_val_batches(
    precomp_val_dir: Path,
    batch_size: int,
    num_val_batches: int,
    seed: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    val을 매번 풀로 돌리면 느리니까,
    시작 시점에 고정된 val 배치 몇 개만 뽑아두고 계속 재사용.
    """
    ds = SLPrecomputedIterableDataset(precomp_val_dir, shuffle_shards=True, shuffle_in_shard=True, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=0, pin_memory=False)

    fixed: list[tuple[torch.Tensor, torch.Tensor]] = []
    it = iter(loader)
    for _ in range(num_val_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        # CPU 텐서로 저장해두기(메모리 절약 + device 이동은 eval 시)
        fixed.append((x.cpu(), y.cpu()))
    return fixed


def train_sl(
    precomp_train_dir: Path,
    out_dir: Path,
    precomp_val_dir: Path | None = None,
    batch_size: int = 128,
    lr: float = 0.01,
    num_steps: int = 2_000_000,
    device: str | None = None,
    save_every: int = 10_000,
    eval_every: int = 2_000,
    num_val_batches: int = 20,   # 20 * 128 = 2560 포지션으로 val 추정
):
    device = pick_device(device)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train dataset
    run_seed = int(time.time())
    dataset = SLPrecomputedIterableDataset(
        precomp_train_dir, 
        shuffle_shards=True, 
        shuffle_in_shard=True, 
        seed=run_seed
        )
    print(f"[data] shuffle seed = {run_seed}")

    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )

    model = PolicyNet().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    ckpt_latest = out_dir / "policy_sl_latest.pt"
    start_step = 0

    if ckpt_latest.exists():
        ckpt = torch.load(ckpt_latest, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_step = int(ckpt.get("step", 0))
        print(f"[resume] loaded {ckpt_latest} (step={start_step})")

    # fixed val batches (optional)
    fixed_val_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    if precomp_val_dir is not None and precomp_val_dir.exists():
        fixed_val_batches = build_fixed_val_batches(
            precomp_val_dir=precomp_val_dir,
            batch_size=batch_size,
            num_val_batches=num_val_batches,
            seed=123, # 고정
        )
        print(f"[val] fixed batches={len(fixed_val_batches)} (each batch={batch_size})")

    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    running_top10 = 0.0
    running_n = 0

    pbar = tqdm(loader, total=num_steps)

    for step, (x, y) in enumerate(pbar, start=start_step + 1):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # train acc (cheap)
        with torch.no_grad():
            acc = topk_accuracy(logits, y, ks=(1, 5, 10))
            bs = x.shape[0]
            running_loss += float(loss.item()) * bs
            running_top1 += acc[1] * bs
            running_top5 += acc[5] * bs
            running_top10 += acc[10] * bs
            running_n += bs

        if step % 100 == 0:
            avg_loss = running_loss / max(running_n, 1)
            avg_t1 = running_top1 / max(running_n, 1)
            avg_t5 = running_top5 / max(running_n, 1)
            avg_t10 = running_top10 / max(running_n, 1)
            pbar.set_description(
                f"step={step} loss={avg_loss:.4f} t1={avg_t1:.3f} t5={avg_t5:.3f} t10={avg_t10:.3f}"
            )
            running_loss = running_top1 = running_top5 = running_top10 = 0.0
            running_n = 0

        # periodic val
        if fixed_val_batches and eval_every and (step % eval_every == 0):
            metrics = evaluate_fixed_batches(model, fixed_val_batches, device=device, ks=(1, 5, 10))
            # tqdm postfix로 표시
            pbar.set_postfix(
                val_loss=f"{metrics['val_loss']:.3f}",
                val_t1=f"{metrics['val_top1']:.3f}",
                val_t5=f"{metrics['val_top5']:.3f}",
                val_t10=f"{metrics['val_top10']:.3f}",
            )

        if save_every and (step % save_every == 0):
            save_checkpoint(out_dir / f"policy_sl_{step}.pt", model, optimizer, step)
            save_checkpoint(ckpt_latest, model, optimizer, step)

        if step >= num_steps:
            break

    save_checkpoint(ckpt_latest, model, optimizer, step)
    print(f"[done] saved {ckpt_latest} (step={step})")


if __name__ == "__main__":
    train_sl(
        precomp_train_dir=Path("precomputed/sl_kgs_alpha48_train"),
        precomp_val_dir=Path("precomputed/sl_kgs_alpha48_val"),
        out_dir=Path("checkpoints/sl_policy"),
        batch_size=128,
        lr=0.01,
        num_steps=2000,
        save_every=500,
        eval_every=200,
        num_val_batches=10,
    )
