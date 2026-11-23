"""Isolated MPS batch benchmark. Never changes production checkpoints/settings."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.10')
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time


def worker(args):
    import statistics
    import numpy as np
    import torch
    import torch.nn.functional as F
    from alphago.models import GoNet, NetConfig
    from alphago.replay import ReplaySampler, atomic_json, sha256
    from alphago.shared_training import training_mode
    from value_source_ab import memory_guard

    torch.set_num_threads(4); torch.manual_seed(20261122)
    torch.mps.set_per_process_memory_fraction(.12)
    device = torch.device('mps')
    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    net = GoNet(NetConfig(**ck['config'])).to(device)
    net.load_state_dict(ck['model'], strict=True)
    training_mode(net)
    groups = [{'params': [p for p in net.parameters() if (p.ndim > 1) == decay],
               'weight_decay': 1e-4 if decay else 0} for decay in (True, False)]
    opt = torch.optim.SGD(groups, lr=.0002, momentum=.9, nesterov=True)
    sampler = ReplaySampler(json.loads(args.pool.read_text()))
    rng = np.random.default_rng(20261122)
    def batch():
        x, pi, z, own, pm, vm, _ = sampler.sample(args.batch, rng, uniform=True, value_fraction=1.)
        return [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in (x, pi, z, own, pm, vm)]
    def step(data):
        x, pi, z, own, pm, vm = data
        p, v, o = net.forward_all(x.float())
        ce = -(pi * F.log_softmax(p, -1)).sum(-1)
        loss = (ce[pm].mean() if pm.any() else ce.sum() * 0) + F.mse_loss(v[vm], z[vm]) + .5 * F.mse_loss(o, own)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True); opt.step()
    results = {}
    fixed = batch()
    for mode in ('compute', 'end_to_end'):
        for _ in range(3):
            step(fixed if mode == 'compute' else batch())
            torch.mps.synchronize()
        timings = []
        for _ in range(3):
            torch.mps.synchronize(); start = time.perf_counter()
            for _ in range(8):
                step(fixed if mode == 'compute' else batch())
                # Production training transfers anew each step; prevent queued work
                # from masquerading as higher throughput.
                torch.mps.synchronize()
            timings.append((time.perf_counter() - start) / 8)
        seconds = statistics.median(timings)
        results[mode] = {'seconds_per_step': seconds, 'positions_per_second': args.batch / seconds,
                         'repeats_seconds_per_step': timings}
        memory_guard()
    result = {'batch': args.batch, 'model_sha256': sha256(args.model), 'config': ck['config'],
              'torch_version': torch.__version__, 'value_fraction': 1., 'sampling': 'uniform_games',
              'mps_memory_limit_fraction': .12, 'mps_allocated_mib': torch.mps.current_allocated_memory() / 1024**2,
              'mps_driver_mib': torch.mps.driver_allocated_memory() / 1024**2, **results}
    atomic_json(args.out, result)


def selfplay_process(pid):
    r = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True)
    return r.returncode == 0 and 'scripts/selfplay_parallel.py' in r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path)
    ap.add_argument('--pool', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--pause-pid', type=int)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--watchdog', type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.watchdog:
        # A separate process resumes production even if the manager is killed.
        time.sleep(600)
        if selfplay_process(args.watchdog):
            os.killpg(args.watchdog, signal.SIGCONT)
        return
    if args.model is None or args.pool is None or args.out is None:
        ap.error('--model, --pool and --out are required')
    if args.worker:
        worker(args)
        return
    from alphago.replay import atomic_json
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    paused, watchdog = False, None
    if args.pause_pid:
        if not selfplay_process(args.pause_pid) or os.getpgid(args.pause_pid) != args.pause_pid:
            raise ValueError('Refusing to pause an unrelated/shared process group')
    def stop(signum, frame): raise KeyboardInterrupt('Benchmark interrupted')
    signal.signal(signal.SIGTERM, stop)
    rows = []
    try:
        if args.pause_pid:
            watchdog = subprocess.Popen([sys.executable, __file__, '--watchdog', str(args.pause_pid)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            os.killpg(args.pause_pid, signal.SIGSTOP); paused = True
            print('Paused self-play without terminating games:', args.pause_pid, flush=True)
        for size in (32, 64, 128, 256, 512):
            target = args.out / f'batch_{size}.json'
            with (args.out / f'batch_{size}.log').open('wb') as log:
                cmd = [sys.executable, __file__, '--worker', '--model', str(args.model), '--pool', str(args.pool),
                       '--out', str(target), '--batch', str(size)]
                try:
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=90)
                    if result.returncode:
                        row = {'batch': size, 'error': 'worker failed; see batch log'}
                    else:
                        row = json.loads(target.read_text())
                except subprocess.TimeoutExpired:
                    row = {'batch': size, 'error': '90-second worker timeout'}
            rows.append(row)
            atomic_json(args.out / 'summary.json', rows)
            print(json.dumps(row), flush=True)
    finally:
        if paused and selfplay_process(args.pause_pid):
            os.killpg(args.pause_pid, signal.SIGCONT)
            print('Resumed self-play:', args.pause_pid, flush=True)
        if watchdog is not None:
            watchdog.terminate(); watchdog.wait()


if __name__ == '__main__':
    main()
