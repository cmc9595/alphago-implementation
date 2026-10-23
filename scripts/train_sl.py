"""Supervised learning of the policy network on human games (AlphaGo's SL stage).

    python scripts/train_sl.py --data data/kgs.npz data/fox_dan.npz
    python scripts/train_sl.py --data data/kgs.npz --steps 2000 --run smoke   # quick check

Resumes automatically from checkpoints/<run>/last.pt. A shared value head is trained
on game outcomes with a small weight; it seeds the later value network.
"""

import os

# This machine has no swap: cap the MPS allocator before torch initialises it.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alphago.data import GameData
from alphago.models import GoNet, NetConfig, pick_device


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def to_device(batch, device):
    feats, targets, z = batch
    x = torch.from_numpy(feats).to(device, non_blocking=True).float()
    return x, torch.from_numpy(targets).to(device), torch.from_numpy(z).to(device)


@torch.no_grad()
def evaluate(net, data, device, batches, batch_size):
    net.eval()
    rng = np.random.default_rng(12345)  # fixed validation sample for comparable numbers
    correct = top5 = n = n_known = 0
    ploss = vloss = 0.0
    for _ in range(batches):
        x, y, z = to_device(data.sample(batch_size, rng, "val", augment=False), device)
        p, v = net(x)
        ploss += F.cross_entropy(p, y, reduction="sum").item()
        known = z != 0
        if known.any():
            vloss += F.mse_loss(v[known], z[known], reduction="sum").item()
            n_known += known.sum().item()
        correct += (p.argmax(1) == y).sum().item()
        top5 += (p.topk(5, 1).indices == y[:, None]).any(1).sum().item()
        n += len(y)
    net.train()
    return {"val_acc": correct / n, "val_top5": top5 / n, "val_policy_loss": ploss / n, "val_value_mse": vloss / max(1, n_known)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--run", default="sl_6x96")
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=300_000, help="~15h for 6x96 at ~1450 samples/s")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--value-weight", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    ckpt_dir = Path("checkpoints") / args.run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "metrics.jsonl"

    data = GameData(args.data)
    print(f"positions: train {data.num_positions('train'):,}  val {data.num_positions('val'):,}")

    torch.manual_seed(args.seed)
    net = GoNet(NetConfig(blocks=args.blocks, channels=args.channels)).to(device)
    decay = [p for n, p in net.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in net.named_parameters() if p.ndim <= 1]
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0}],
        lr=args.lr, momentum=0.9, nesterov=True,
    )
    step = 0
    last = ckpt_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        print(f"resumed from {last} at step {step}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "config": vars(net.cfg), "args": vars(args)}, tmp)
        tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint

    stream = data.iterate(args.batch, seed=args.seed + step)
    net.train()
    t0, seen, run_p, run_v, run_acc = time.perf_counter(), 0, 0.0, 0.0, 0.0
    while step < args.steps:
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, z = to_device(next(stream), device)
        p, v = net(x)
        policy_loss = F.cross_entropy(p, y)
        known = z != 0
        value_loss = F.mse_loss(v[known], z[known]) if known.any() else v.sum() * 0
        loss = policy_loss + args.value_weight * value_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        seen += len(y)
        run_p += policy_loss.item()
        run_v += value_loss.item()
        run_acc += (p.argmax(1) == y).float().mean().item()

        if step % 100 == 0:
            dt = time.perf_counter() - t0
            rec = {"step": step, "lr": round(lr, 5), "policy_loss": run_p / 100, "value_mse": run_v / 100,
                   "train_acc": run_acc / 100, "samples_per_s": round(seen / dt)}
            run_p = run_v = run_acc = 0.0
            if step % args.eval_every == 0 or step == args.steps:
                rec.update(evaluate(net, data, device, args.eval_batches, args.batch))
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            t0, seen = time.perf_counter(), 0
        if step % args.save_every == 0 or step == args.steps:
            save(last)
        if step % 50_000 == 0:
            save(ckpt_dir / f"step{step}.pt")
    save(ckpt_dir / "final.pt")
    print("done")


if __name__ == "__main__":
    main()
