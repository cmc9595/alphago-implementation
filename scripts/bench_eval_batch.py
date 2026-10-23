"""Network evaluation throughput by batch size (features + forward + copy back)."""
import time, torch, numpy as np
from alphago import Board
from alphago.gtp import load_net
from alphago.mcts import NetEvaluator
dev = torch.device('mps'); torch.mps.set_per_process_memory_fraction(.2)
ev = NetEvaluator(load_net('checkpoints/shared_supervised_6x96_20261018/best.pt', dev), device=dev)
b = Board(19, 7.5, superko=True)
for m in [72, 288, 60, 300, 120, 241]: b.play(m)
for n in [8, 32, 64, 128, 256, 512]:
    boards = [b.copy() for _ in range(n)]
    for _ in range(3): ev(boards)
    reps = max(5, 2048 // n); t = time.perf_counter()
    for _ in range(reps): ev(boards)
    torch.mps.synchronize(); s = time.perf_counter() - t
    print(f'batch {n:4d}: {reps*n/s:8.0f} positions/s  {1000*s/reps:7.1f} ms/call', flush=True)
