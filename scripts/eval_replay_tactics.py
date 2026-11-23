"""Read-only T5 diagnostic; not a comprehensive life-and-death benchmark."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.10')
import argparse
import json
from pathlib import Path

import torch

from alphago import Board
from alphago.gtp import action_to_vertex, load_net, vertex_to_action
from alphago.mcts import MCTS, NetEvaluator
from alphago.replay import atomic_json, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('models', nargs='+', type=Path)
    ap.add_argument('--position', type=Path, default=Path('runs/diagnostics/t5_20261019/position.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    torch.set_num_threads(4); torch.mps.set_per_process_memory_fraction(.12)
    position = json.loads(args.position.read_text())
    board = Board(19, position.get('komi', 7.5), superko=True)
    for v in position['history']: board.play(vertex_to_action(v, 19))
    rows = []
    for path in args.models:
        evaluator = NetEvaluator(load_net(str(path), torch.device('mps')))
        _, value = evaluator([board])
        root = MCTS(evaluator, playouts=1600, batch=32).search(board)
        top = [{'move': action_to_vertex(int(root.moves[i]), 19), 'visits': int(root.N[i]),
                'q': float(root.W[i] / root.N[i])} for i in root.N.argsort()[::-1][:5] if root.N[i] > 0]
        rows.append({'model': str(path), 'model_sha256': sha256(path), 'value_side_to_move': float(value[0]), 'top': top})
        del evaluator, root
        torch.mps.empty_cache()
    atomic_json(args.out, {'position_sha256': sha256(args.position), 'visits': 1600, 'results': rows,
                           'note': 'Single diagnostic, not a pass/fail life-and-death suite'})


if __name__ == '__main__':
    main()
