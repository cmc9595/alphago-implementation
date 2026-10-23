"""Train the value network (optionally with the KataGo-style ownership head).

    python scripts/train_value.py --run value_own --ownership-weight 1.5
    python scripts/train_value.py --run value_plain --ownership-weight 0

The trunk starts from the final RL network (its features already encode the game); the policy
head is dropped. Evaluation uses held-out games: MSE and winner accuracy by source and by phase.
Resumes from checkpoints/<run>/last.pt.
"""

import os

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

from alphago.models import GoNet, NetConfig, pick_device
from alphago.value_data import ValueData, load_gen, load_human, load_rl

PHASES = [(0, 30, "opening"), (30, 80, "early"), (80, 150, "middle"), (150, 1000, "late")]


def lr_at(step, total, base, warmup=500):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


@torch.no_grad()
def evaluate(net, data, device, rows):
    net.eval()
    v_all, z_all, own_hit, own_n = [], [], 0, 0
    for i in range(0, len(rows), 512):
        feats, z, own = data.batch_from_index(rows[i : i + 512])
        _, v, o = net.forward_all(torch.from_numpy(feats).to(device).float())
        v_all.append(v.cpu().numpy())
        z_all.append(z)
        if o is not None:
            o = o.cpu().numpy()
            labelled = own != 0
            own_hit += ((np.sign(o) == own) & labelled).sum()
            own_n += labelled.sum()
    net.train()
    v, z = np.concatenate(v_all), np.concatenate(z_all)
    out = {}

    def stats(mask, key):
        m = mask & (z != 0)
        if m.any():
            out[key] = {"n": int(m.sum()), "mse": round(float(((v[m] - z[m]) ** 2).mean()), 4),
                        "winner_acc": round(float((np.sign(v[m]) == z[m]).mean()), 4)}

    stats(np.ones(len(z), bool), "all")
    for si, src in enumerate(data.sources):
        stats(rows[:, 0] == si, src.name)
    for lo, hi, name in PHASES:
        stats((rows[:, 2] >= lo) & (rows[:, 2] < hi), name)
    gen_id = next((i for i, s in enumerate(data.sources) if s.name == "gen"), None)
    if gen_id is not None:  # the AlphaGo-comparable number: held-out generated positions
        stats(rows[:, 0] == gen_id, "gen_test")
    if own_n:
        out["ownership_point_acc"] = round(float(own_hit / own_n), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="value_own")
    ap.add_argument("--init", default="checkpoints/rl_6x96/final.pt", help="'none' = random init")
    ap.add_argument("--blocks", type=int, default=None, help="with --init none")
    ap.add_argument("--channels", type=int, default=None, help="with --init none")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ownership-weight", type=float, default=1.5, help="0 = no ownership head (AlphaGo-style)")
    ap.add_argument("--no-human", action="store_true")
    ap.add_argument("--gen-only", action="store_true",
                    help="AlphaGo-style: only RL-generated data (data/value/gen + data/value/gen_rlcont)")
    ap.add_argument("--no-rl-archive", action="store_true")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-positions", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    run_dir = Path("checkpoints") / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.jsonl"

    sources = [load_gen()]
    if args.gen_only:
        sources.append(load_gen("data/value/gen_rlcont/batch_*.npz", name="gen2"))
    elif not args.no_rl_archive:
        sources.append(load_rl())
    if not args.gen_only and not args.no_human and Path("data/value/human_kgs.npz").exists():
        sources.append(load_human())
    data = ValueData(sources)
    print({s.name: len(s.offsets) - 1 for s in sources}, "games;",
          f"train {data.size('train'):,} test {data.size('test'):,} positions", flush=True)
    eval_rows = data.index["test"][np.random.default_rng(1).permutation(data.size("test"))[: args.eval_positions]]

    own_on = args.ownership_weight > 0
    if args.init == "none":
        cfg = NetConfig(blocks=args.blocks, channels=args.channels, policy=False, ownership=own_on)
        net = GoNet(cfg).to(device)
        print("random init", args.blocks, "x", args.channels, flush=True)
    else:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        cfg = NetConfig(**{**init["config"], "policy": False, "ownership": own_on})
        net = GoNet(cfg).to(device)
        state = {k: v for k, v in init["model"].items() if not k.startswith("policy_")}
        missing, _ = net.load_state_dict(state, strict=False)
        print("initialised from", args.init, "| new parameters:", missing, flush=True)

    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim <= 1]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.weight_decay},
                           {"params": no_decay, "weight_decay": 0}], lr=args.lr, momentum=0.9, nesterov=True)
    step = 0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        print("resumed at step", step, flush=True)

    def save(path, extra=None):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "step": step,
                    "config": vars(cfg), "args": vars(args), **(extra or {})}, tmp)
        tmp.replace(path)

    stream = data.iterate(args.batch, seed=args.seed + step)
    net.train()
    t0, run_v, run_o = time.time(), 0.0, 0.0
    best = None
    while step < args.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        feats, z, own = next(stream)
        x = torch.from_numpy(feats).to(device).float()
        zt = torch.from_numpy(z).to(device)
        _, v, o = net.forward_all(x)
        known = zt != 0
        vloss = F.mse_loss(v[known], zt[known]) if known.any() else v.sum() * 0
        loss = vloss
        if own_on:
            ot = torch.from_numpy(own).to(device)
            oloss = F.mse_loss(o, ot)
            loss = loss + args.ownership_weight * oloss
            run_o += oloss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        run_v += vloss.item()
        if step % 200 == 0:
            rec = {"step": step, "lr": round(opt.param_groups[0]["lr"], 5), "train_value_mse": round(run_v / 200, 4),
                   "samples_per_s": round(200 * args.batch / (time.time() - t0))}
            if own_on:
                rec["train_ownership_mse"] = round(run_o / 200, 4)
            run_v = run_o = 0.0
            if step % args.eval_every == 0 or step == args.steps:
                rec["test"] = evaluate(net, data, device, eval_rows)
                mse = rec["test"]["all"]["mse"]
                if best is None or mse < best:
                    best = mse
                    save(run_dir / "best.pt", {"test": rec["test"]})
            print(json.dumps(rec), flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            t0 = time.time()
        if step % 1000 == 0:
            save(last)
    save(run_dir / "final.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
