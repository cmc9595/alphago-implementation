"""Compare checkpoint values on identical held-out positions and board symmetries."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.3')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.25')
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from alphago.gtp import load_net
from alphago.value_data import ValueData, load_gen, load_rl, load_human
from alphago.data import transform

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--positions', type=int, default=12000)
    ap.add_argument('--out', default='runs/value_comparison.json')
    args = ap.parse_args()
    data = ValueData([load_gen(), load_rl(), load_human()])
    rows = data.index['test'][np.random.default_rng(1).permutation(data.size('test'))[:args.positions]]
    paths = {'sl': 'checkpoints/sl_6x96/final.pt', 'rl': 'checkpoints/rl_6x96/final.pt',
             'value_own': 'checkpoints/value_own/best.pt', 'value_plain': 'checkpoints/value_plain/best.pt'}
    nets = {k: load_net(p) for k,p in paths.items()}
    values = {k: [] for k in nets}
    targets = []
    sym_features = None
    for i in range(0, len(rows), 128):
        feats, z, _ = data.batch_from_index(rows[i:i+128])
        if sym_features is None:
            sym_features = feats[:64].copy()
        targets.append(z)
        for k, net in nets.items():
            x = torch.from_numpy(feats).to(next(net.parameters()).device).float()
            values[k].append(net(x)[1].cpu().numpy())
    z = np.concatenate(targets)
    masks = {'all': z != 0}
    for i, src in enumerate(data.sources):
        masks[src.name] = (rows[:,0] == i) & (z != 0)
    for lo,hi,name in [(0,30,'opening'),(30,80,'early'),(80,150,'middle'),(150,1000,'late')]:
        masks[name] = (rows[:,2] >= lo) & (rows[:,2] < hi) & (z != 0)
    out = {'positions': len(rows), 'models': {}}
    for k, net in nets.items():
        v = np.concatenate(values[k])
        stats = {}
        for name, mask in masks.items():
            p, t = (v[mask]+1)/2, (z[mask]+1)/2
            ece = 0.0
            for b in range(10):
                m = (p >= b/10) & (p < (b+1)/10 if b<9 else p<=1)
                if m.any(): ece += m.mean()*abs(p[m].mean()-t[m].mean())
            stats[name] = {'n': int(mask.sum()), 'mse': float(np.mean((v[mask]-z[mask])**2)),
                           'winner_acc': float(np.mean(np.sign(v[mask])==z[mask])),
                           'calibration_ece': float(ece)}
        sym_values = []
        for s in range(8):
            x = torch.from_numpy(np.ascontiguousarray(transform(sym_features,s))).to(next(net.parameters()).device).float()
            sym_values.append(net(x)[1].cpu().numpy())
        sym = np.stack(sym_values)
        stats['symmetry_win_probability_range_mean'] = float(np.ptp((sym+1)/2,axis=0).mean())
        stats['symmetry_win_probability_range_max'] = float(np.ptp((sym+1)/2,axis=0).max())
        out['models'][k] = stats
        print(k, json.dumps(stats), flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

if __name__ == '__main__': main()
