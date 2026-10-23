"""Week-0 hardware benchmark: engine feature throughput and NN train/inference speed.

Each measurement runs in its own subprocess with a GPU memory cap, so a large config
cannot accumulate allocator caches and take the whole machine down (no swap here).

    python scripts/bench.py                       # everything, one subprocess per case
    python scripts/bench.py --case engine         # a single case in-process
    python scripts/bench.py --case mps:6x96:fp32
"""

import argparse
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from alphago import Board, features_batch
from alphago.models import GoNet, NetConfig

CONFIGS = ["6x96", "8x128", "10x128", "12x192"]  # blocks x channels; 12x192 ~ AlphaGo-sized
MEM_LIMIT_GB = 8


def random_positions(count, n=19, seed=0):
    rng = random.Random(seed)
    boards = []
    while len(boards) < count:
        b = Board(n)
        for _ in range(rng.randint(10, 250)):
            moves = np.flatnonzero(b.sensible_mask()[:-1])
            if len(moves) == 0:
                break
            b.play(int(rng.choice(list(moves))))
        boards.append(b)
    return boards


def bench_engine(boards):
    t = time.perf_counter()
    for b in boards:
        b.features()
    single = len(boards) / (time.perf_counter() - t)
    t = time.perf_counter()
    features_batch(boards)
    batch = len(boards) / (time.perf_counter() - t)

    # Raw play speed: random sensible-move games, the inner loop of RL self-play minus the network.
    t, moves = time.perf_counter(), 0
    rng = np.random.default_rng(0)
    for _ in range(20):
        b = Board(19)
        while not b.game_over and b.move_number < 400:
            mask = b.sensible_mask()
            legal = np.flatnonzero(mask[:-1])
            b.play(int(rng.choice(legal)) if len(legal) else b.pass_action)
            moves += 1
    play = moves / (time.perf_counter() - t)
    return {"features_per_s_1thread": round(single), "features_per_s_all_threads": round(batch),
            "random_moves_per_s": round(play)}


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench_torch(device, blocks, channels, train_batch=256, steps=20, dtype=None):
    torch.manual_seed(0)
    net = GoNet(NetConfig(blocks=blocks, channels=channels)).to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    x = torch.randn(train_batch, net.cfg.in_planes, 19, 19, device=device)
    y = torch.randint(0, 362, (train_batch,), device=device)
    z = torch.rand(train_batch, device=device) * 2 - 1

    def step():
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            p, v = net(x)
            loss = F.cross_entropy(p, y) + F.mse_loss(v.float(), z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    net.train()
    for _ in range(3):
        step()
    sync(device)
    t = time.perf_counter()
    for _ in range(steps):
        step()
    sync(device)
    train = train_batch * steps / (time.perf_counter() - t)

    net.eval()
    infer = {}
    with torch.inference_mode():
        for bs in (16, 256):
            xi = torch.randn(bs, net.cfg.in_planes, 19, 19, device=device)
            for _ in range(3):
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    net(xi)
            sync(device)
            reps = 40 if bs == 16 else 10
            t = time.perf_counter()
            for _ in range(reps):
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    p, _ = net(xi)
                p.cpu()  # include the device->host copy MCTS will need
            infer[f"infer_pos_per_s_b{bs}"] = round(bs * reps / (time.perf_counter() - t))
    return {"train_samples_per_s": round(train), **infer}


def bench_mlx(blocks, channels, train_batch=256, steps=20):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    class Block(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.c1, self.b1 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm(c)
            self.c2, self.b2 = nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm(c)

        def __call__(self, x):
            y = nn.relu(self.b1(self.c1(x)))
            return nn.relu(x + self.b2(self.c2(y)))

    class Net(nn.Module):
        def __init__(self, planes, c, b):
            super().__init__()
            self.stem = nn.Conv2d(planes, c, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm(c)
            self.blocks = [Block(c) for _ in range(b)]
            self.pconv, self.pbn = nn.Conv2d(c, 2, 1, bias=False), nn.BatchNorm(2)
            self.pfc = nn.Linear(2 * 361, 362)
            self.vconv, self.vbn = nn.Conv2d(c, 1, 1, bias=False), nn.BatchNorm(1)
            self.vfc1, self.vfc2 = nn.Linear(361, 256), nn.Linear(256, 1)

        def __call__(self, x):  # NHWC
            h = nn.relu(self.bn(self.stem(x)))
            for blk in self.blocks:
                h = blk(h)
            p = self.pfc(nn.relu(self.pbn(self.pconv(h))).reshape(x.shape[0], -1))
            v = mx.tanh(self.vfc2(nn.relu(self.vfc1(nn.relu(self.vbn(self.vconv(h))).reshape(x.shape[0], -1)))))
            return p, v.squeeze(-1)

    from alphago import NUM_FEATURES

    net = Net(NUM_FEATURES, channels, blocks)
    mx.set_memory_limit(MEM_LIMIT_GB << 30)
    mx.set_cache_limit(1 << 30)
    mx.eval(net.parameters())
    opt = optim.SGD(learning_rate=0.01, momentum=0.9)
    x = mx.random.normal((train_batch, 19, 19, NUM_FEATURES))
    y = mx.random.randint(0, 362, (train_batch,))
    z = mx.random.uniform(-1, 1, (train_batch,))

    def loss_fn(model, x, y, z):
        p, v = model(x)
        return nn.losses.cross_entropy(p, y).mean() + ((v - z) ** 2).mean()

    grad_fn = nn.value_and_grad(net, loss_fn)

    def step():
        loss, grads = grad_fn(net, x, y, z)
        opt.update(net, grads)
        mx.eval(net.parameters(), opt.state, loss)

    net.train()
    for _ in range(3):
        step()
    t = time.perf_counter()
    for _ in range(steps):
        step()
    train = train_batch * steps / (time.perf_counter() - t)

    net.eval()
    infer = {}
    for bs in (16, 256):
        xi = mx.random.normal((bs, 19, 19, NUM_FEATURES))
        for _ in range(3):
            mx.eval(net(xi))
        reps = 40 if bs == 16 else 10
        t = time.perf_counter()
        for _ in range(reps):
            p, v = net(xi)
            np.array(p)
        infer[f"infer_pos_per_s_b{bs}"] = round(bs * reps / (time.perf_counter() - t))
    return {"train_samples_per_s": round(train), **infer}


def run_case(case):
    if case == "engine":
        return bench_engine(random_positions(2000))
    backend, cfg, prec = case.split(":")
    blocks, ch = map(int, cfg.split("x"))
    if backend == "mlx":
        return bench_mlx(blocks, ch)
    device = torch.device(backend)
    dtype = torch.bfloat16 if prec == "bf16" else None
    return bench_torch(device, blocks, ch, dtype=dtype, steps=5 if backend == "cpu" else 20)


def all_cases(skip_mlx, skip_cpu):
    cases = ["engine"]
    for cfg in CONFIGS:
        cases += [f"mps:{cfg}:fp32", f"mps:{cfg}:bf16"]
        if not skip_mlx:
            cases.append(f"mlx:{cfg}:fp32")
    if not skip_cpu:
        cases.append(f"cpu:{CONFIGS[0]}:fp32")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", help="run a single case in-process and print JSON")
    ap.add_argument("--skip-mlx", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--out", default="bench_results.json")
    args = ap.parse_args()

    if args.case:
        try:
            r = run_case(args.case)
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {e}"[:300]}
        r["max_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
        print("RESULT " + json.dumps(r), flush=True)
        return

    results = {"machine": platform.platform(), "torch": torch.__version__}
    env = dict(os.environ, PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.3", PYTORCH_MPS_LOW_WATERMARK_RATIO="0.25")
    for case in all_cases(args.skip_mlx, args.skip_cpu):
        proc = subprocess.run([sys.executable, __file__, "--case", case], env=env,
                              capture_output=True, text=True, timeout=900)
        lines = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
        results[case] = json.loads(lines[-1][7:]) if lines else {"error": proc.stderr[-300:]}
        print(case, json.dumps(results[case]), flush=True)
        with open(args.out, "w") as f:  # save after every case in case something dies
            json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
