"""Small, resumable self-play-only updates using game-uniform replay."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.10')
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

from alphago.models import GoNet, NetConfig, pick_device
from alphago.replay import ReplaySampler, atomic_json, sha256
from alphago.shared_training import training_mode
from value_source_ab import memory_guard


def atomic_checkpoint(path, body):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as stream: torch.save(body, stream)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


@torch.no_grad()
def evaluate(net, sampler, device):
    net.eval()
    ce = hits = mse = own_mse = game_mse = 0.
    n = policy_n = 0
    for row in sampler.pool['validation']:
        d = sampler.load(row)
        errors = []
        for start in range(0, len(d['z']), 64):
            x = np.unpackbits(d['x'][start:start + 64], axis=1, count=34)
            x = torch.from_numpy(x).to(device).float()
            pi = torch.from_numpy(d['pi'][start:start + 64].astype(np.float32)).to(device)
            z = torch.from_numpy(d['z'][start:start + 64]).to(device)
            own = torch.from_numpy(d['own'][start:start + 64].astype(np.float32)).to(device)
            p, v, o = net.forward_all(x)
            err = (v - z).square()
            errors.extend(err.cpu().tolist())
            mse += float(err.sum()); n += len(z)
            own_mse += float((o - own).square().mean((1, 2)).sum())
            if row['policy']:
                ce += float(-(pi * F.log_softmax(p, -1)).sum())
                hits += float((p.argmax(1) == pi.argmax(1)).sum()); policy_n += len(z)
        game_mse += float(np.mean(errors))
    if not n or not policy_n: raise ValueError('Empty replay validation targets')
    return {'selfplay_policy_ce': ce / policy_n, 'selfplay_top1_match': hits / policy_n,
            'selfplay_value_mse': mse / n, 'selfplay_ownership_mse': own_mse / n,
            'mean_game_mse': game_mse / len(sampler.pool['validation']), 'selfplay_val_positions': n}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--init', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=.0002)
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--sampling', choices=['stratified', 'uniform'], default='stratified',
                    help='uniform: choose games uniformly from the entire active pool')
    ap.add_argument('--value-fraction', type=float, default=.5,
                    help='fraction of sampled positions supervised by terminal outcome; 1 uses every row')
    args = ap.parse_args()
    if min(args.steps, args.batch, args.eval_every) <= 0 or args.lr <= 0: ap.error('Invalid training budget')
    if not 0 < args.value_fraction <= 1: ap.error('--value-fraction must be in (0, 1]')
    args.out.mkdir(parents=True, exist_ok=True)
    signature = {'pool_sha256': sha256(args.pool), 'init_sha256': sha256(args.init),
                 'steps': args.steps, 'batch': args.batch, 'lr': args.lr, 'seed': args.seed,
                 'latest_fraction': .5, 'value_fraction': args.value_fraction, 'search_mix': 0,
                 'optimizer': 'SGD momentum .9, nesterov, decay 1e-4 on matrices, constant LR'}
    if args.sampling == 'uniform':
        signature.update(sampling='uniform_games', latest_fraction=None)
    final, last = args.out / 'final.pt', args.out / 'last.pt'
    if final.exists():
        ck = torch.load(final, map_location='cpu', weights_only=False)
        if ck['replay_signature'] != signature: raise ValueError('Existing final checkpoint belongs to another run')
        print('Already completed', final, flush=True)
        return
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    device = pick_device()
    if device.type != 'mps': raise RuntimeError('MPS unavailable')
    torch.mps.set_per_process_memory_fraction(.12)
    pool = json.loads(args.pool.read_text())
    sampler = ReplaySampler(pool)
    ck = torch.load(last if last.exists() else args.init, map_location='cpu', weights_only=False)
    if last.exists() and ck['replay_signature'] != signature: raise ValueError('Resume configuration changed')
    net = GoNet(NetConfig(**ck['config'])).to(device)
    if not net.cfg.ownership or net.cfg.score: raise ValueError('Pilot expects policy/value/ownership and no score head')
    net.load_state_dict(ck['model'], strict=True)
    groups = [{'params': [p for p in net.parameters() if (p.ndim > 1) == decay],
               'weight_decay': 1e-4 if decay else 0} for decay in (True, False)]
    opt = torch.optim.SGD(groups, lr=args.lr, momentum=.9, nesterov=True)
    if 'optimizer' in ck:
        opt.load_state_dict(ck['optimizer'])
    for group in opt.param_groups: group['lr'] = args.lr
    rng = np.random.default_rng(args.seed)
    history, exposures = [], Counter()
    start, elapsed = 0, 0.
    base_total_updates = ck.get('total_updates', ck.get('step', 0))
    if args.sampling == 'uniform' and 'numpy_rng' in ck:
        rng.bit_generator.state = ck['numpy_rng']
        torch.set_rng_state(ck['torch_rng'])
        if ck.get('mps_rng') is not None: torch.mps.set_rng_state(ck['mps_rng'])
    if last.exists():
        start, elapsed, history = ck['step'], ck['seconds'], ck['history']
        rng.bit_generator.state = ck['numpy_rng']
        torch.set_rng_state(ck['torch_rng'])
        if ck.get('mps_rng') is not None: torch.mps.set_rng_state(ck['mps_rng'])
        exposures.update(ck['game_exposures'])
        base_total_updates = ck.get('base_total_updates', 0)
    info = {'args': {**signature, 'human_batch': 0, 'gen_batch': 0}, 'games': len(pool['train']),
            'train_positions': sum(r['positions'] for r in pool['train']),
            'val_positions': sum(r['positions'] for r in pool['validation']),
            'latest_games': len(sampler.latest), 'history_games': len(sampler.history),
            'optimizer_carried_from_champion': 'optimizer' in ck}
    info.update(optimizer_carried_from_input='optimizer' in ck, base_total_updates=base_total_updates)
    if args.sampling == 'uniform':
        info.pop('optimizer_carried_from_champion', None)
    old_info = args.out / 'info.json'
    if last.exists() and old_info.exists():
        info = json.loads(old_info.read_text())
    else:
        info['baseline'] = evaluate(net, sampler, device)
        atomic_json(old_info, info)
    print(json.dumps(info), flush=True)
    memory_guard(); training_mode(net); t0 = time.monotonic()
    for step in range(start + 1, args.steps + 1):
        x, pi, z, own, pm, vm, identities = sampler.sample(args.batch, rng, uniform=args.sampling == 'uniform',
                                                         value_fraction=args.value_fraction)
        exposures.update(h for h, _ in identities)
        x, pi, z, own, pm, vm = [torch.from_numpy(np.ascontiguousarray(a)).to(device)
                                  for a in (x, pi, z, own, pm, vm)]
        p, v, o = net.forward_all(x.float())
        ce = -(pi * F.log_softmax(p, -1)).sum(-1)
        policy_loss = ce[pm].mean() if pm.any() else ce.sum() * 0
        value_loss = F.mse_loss(v[vm], z[vm]) if vm.any() else v.sum() * 0
        own_loss = F.mse_loss(o, own)
        loss = policy_loss + value_loss + .5 * own_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True); opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            rec = {'step': step, 'steps': args.steps, 'seconds': elapsed + time.monotonic() - t0,
                   'loss': float(loss.detach()), 'total_updates': base_total_updates + step,
                   **evaluate(net, sampler, device)}
            history.append(rec); training_mode(net)
            body = {'model': net.state_dict(), 'config': vars(net.cfg), 'optimizer': opt.state_dict(),
                    'step': step, 'seconds': rec['seconds'], 'history': history, 'replay_signature': signature,
                    'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                    'mps_rng': torch.mps.get_rng_state(), 'game_exposures': dict(exposures),
                    'base_total_updates': base_total_updates, 'total_updates': base_total_updates + step}
            atomic_checkpoint(last, body)
            atomic_json(args.out / 'history.json', history)
            counts = list(exposures.values())
            atomic_json(args.out / 'sampling.json', {'draws': sum(counts), 'unique_games': len(counts),
                                                    'max_draws_per_game': max(counts),
                                                    'mean_draws_per_sampled_game': float(np.mean(counts))})
            print(json.dumps(rec), flush=True); memory_guard()
    # Reload last atomically persisted checkpoint; handles resume at the final step too.
    body = torch.load(last, map_location='cpu', weights_only=False)
    atomic_checkpoint(final, body)
    print('saved', final, flush=True)


if __name__ == '__main__':
    main()
