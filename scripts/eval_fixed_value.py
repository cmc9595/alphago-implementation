"""Compare direct value predictions on frozen whole-game holdout positions."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.10')
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from alphago.gtp import load_net
from alphago.models import pick_device


@torch.no_grad()
def evaluate(model, folder, manifest, batch, device):
    net = load_net(str(model), device).eval()
    per_game, squared_sum, positions = [], 0., 0
    for row in manifest['test_games']:
        with np.load(folder / 'test' / row['file']) as d:
            digest = hashlib.sha256(np.asarray(d['moves'], dtype='<i2').tobytes()).hexdigest()
            if digest != row['sha256_moves']:
                raise ValueError(f"Holdout changed: {row['file']}")
            score = float(d['score'])
            if bool(d['capped']) or not bool(d['judged']) or not np.isfinite(score) or score == 0:
                raise ValueError(f"Invalid holdout game: {row['file']}")
            idx = np.asarray(row['plies'], dtype=np.int64)
            if not len(idx): raise ValueError('Empty evaluation game')
            x = d['features'][idx]
            z = np.sign(score) * np.where(idx % 2 == 0, 1., -1.)
        errors = []
        for start in range(0, len(idx), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[start:start + batch])).to(device).float()
            _, v = net(xb)
            prediction = v.cpu().numpy()
            if not np.isfinite(prediction).all(): raise ValueError('Nonfinite prediction')
            errors.extend(((prediction - z[start:start + batch]) ** 2).tolist())
        squared_sum += sum(errors); positions += len(errors)
        per_game.append({'file': row['file'], 'mse': float(np.mean(errors)), 'positions': len(errors)})
    if not positions: raise ValueError('Empty holdout')
    return {'model': str(model), 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
            'games': len(per_game), 'positions': positions, 'position_mse': squared_sum / positions,
            'mean_game_mse': float(np.mean([g['mse'] for g in per_game])), 'per_game': per_game}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('models', nargs='+', type=Path)
    ap.add_argument('--dataset', type=Path, default=Path('runs/datasets/round6_fixed1000'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--device', choices=['cpu', 'mps'], default=None)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    if args.batch < 1: ap.error('--batch must be positive')
    torch.set_num_threads(4)
    device = torch.device(args.device) if args.device else pick_device()
    if device.type == 'mps': torch.mps.set_per_process_memory_fraction(.12)
    raw = (args.dataset / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    results = []
    for model in args.models:
        rec = evaluate(model, args.dataset, manifest, args.batch, device)
        results.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != 'per_game'}), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump({'manifest_sha256': hashlib.sha256(raw).hexdigest(), 'results': results}, stream, indent=2)


if __name__ == '__main__':
    main()
