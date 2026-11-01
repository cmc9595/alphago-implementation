"""Fine-tune the shared policy/value/ownership net on MCTS self-play (one update per position).

Self-play rows (400-visit moves only) supervise all heads in one forward pass:
  policy    <- root visit distribution (all rows)
  ownership <- final GNU Go ownership (all rows)
  value     <- final GNU Go result (only a fixed random 1/20 of rows; masked elsewhere)
Small anchor batches keep old knowledge: human moves (policy only) and the old generated
value data (value + ownership). Validation is split by whole game.
"""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.10')
import argparse, glob, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from alphago.data import GameData, symmetry_tables, transform
from alphago.models import GoNet, NetConfig, pick_device, SCORE_SCALE
from alphago.shared_training import training_mode
from alphago.value_data import ValueData, load_gen
from value_source_ab import write_json, memory_guard


class SelfPlay:
    """Keeps only train_mask (400-visit) positions in memory."""

    def __init__(self, run_dirs, value_fraction, seed, val_mod=10, search_mix=0.0, dedup=False):
        feats, pis, zs, owns, game_ids, svs, scs = [], [], [], [], [], [], []
        self.rejected = 0
        paths = [(k, p) for k, d in enumerate(run_dirs) for p in sorted(glob.glob(str(d / 'game_*.npz')))]
        seen = set(); self.duplicates = 0
        for k, path in paths:
            with np.load(path) as d:
                if dedup:
                    key = d['moves'].tobytes()
                    if key in seen:
                        self.duplicates += 1; continue
                    seen.add(key)
                if bool(d['capped']) or not bool(d['judged']) or not np.isfinite(d['score']) or d['score'] == 0:
                    self.rejected += 1; continue
                idx = np.flatnonzero(d['train_mask'])
                side = np.where(idx % 2 == 0, 1., -1.).astype(np.float32)  # +1 = black to move
                f_ = d['features'][idx]
                if f_.max() > 1: raise ValueError('non-binary feature plane; bit-packing would lose data')
                feats.append(np.packbits(f_, axis=1)); pis.append(d['policy'][idx].astype(np.float16))  # 8x / 2x less RAM
                zs.append(np.sign(d['score']).astype(np.float32) * side)
                svs.append(d['search_values'][idx].astype(np.float32))  # root Q, side-to-move view
                scs.append(np.float32(d['score']) * side)  # final margin, side-to-move view
                owns.append(d['ownership'].reshape(1, 19, 19).astype(np.int8) * side[:, None, None].astype(np.int8))
                game_ids.append(np.full(len(idx), k * 10**9 + int(d['seed'])))  # unique across runs
        self.planes = feats[0].shape[1] * 8 if feats else 0
        self.n_planes = None
        self.x = np.concatenate(feats); del feats; self.pi = np.concatenate(pis)
        self.z = np.concatenate(zs); self.own = np.concatenate(owns); self.score = np.concatenate(scs)
        # Training value target: blend of final outcome and the MCTS root value (KataGo/Lc0-style).
        # Evaluation keeps the pure outcome so numbers stay comparable across runs.
        self.zt = ((1 - search_mix) * self.z + search_mix * np.concatenate(svs)).astype(np.float32)
        games = np.concatenate(game_ids)
        uniq = np.unique(games); val_games = set(uniq[np.random.default_rng(seed).permutation(len(uniq))[:len(uniq) // val_mod]])
        is_val = np.isin(games, list(val_games))
        self.train, self.val = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        self.value_mask = np.random.default_rng(seed + 1).random(len(self.z)) < value_fraction
        self.games = len(uniq); self.perm = symmetry_tables(19)

    def batch(self, rows, rng=None):
        x = np.unpackbits(self.x[rows], axis=1, count=34)
        pi = self.pi[rows].astype(np.float32)
        own, z, vm = self.own[rows].astype(np.float32), self.z[rows], self.value_mask[rows]
        if rng is not None:
            z = self.zt[rows]  # training batches use the blended target
        if rng is not None:
            syms = rng.integers(0, 8, len(rows))
            for s in range(1, 8):
                sel = np.flatnonzero(syms == s)
                if len(sel):
                    x[sel] = transform(x[sel], s); own[sel] = transform(own[sel], s)
                    old = pi[sel].copy(); pi[np.ix_(sel, self.perm[s])] = old
        return x, pi, z, own, vm, self.score[rows]


def to(device, *arrays):
    return [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays]


@torch.no_grad()
def evaluate(net, sp, human_eval, gen_eval, device, batch=128):
    net.eval()
    ce = acc = mse = own_mse = score_abs = 0.; n = 0
    for i in range(0, len(sp.val), batch):
        x, pi, z, own, _, sc = sp.batch(sp.val[i:i + batch])
        x, pi, z, own, sc = to(device, x, pi, z, own, sc)
        p, v, o, s_ = net.forward_heads(x.float())
        ce += float(-(pi * F.log_softmax(p, -1)).sum()); acc += float((p.argmax(1) == pi.argmax(1)).sum())
        mse += float(((v - z) ** 2).sum()); own_mse += float(((o - own) ** 2).mean((1, 2)).sum()); n += len(z)
        if s_ is not None: score_abs += float((s_ * SCORE_SCALE - sc).abs().sum())
    out = {'selfplay_policy_ce': ce / n, 'selfplay_top1_match': acc / n, 'selfplay_value_mse': mse / n,
           'selfplay_ownership_mse': own_mse / n, 'selfplay_val_positions': n}
    if net.cfg.score: out['selfplay_score_mae'] = score_abs / n
    hx, hy = human_eval; gx, gz, _, gsc = gen_eval
    hit = 0; vs = []; ss = []
    for i in range(0, len(hy), batch):
        p, _, _ = net.forward_all(torch.from_numpy(hx[i:i + batch]).to(device).float())
        hit += int((p.argmax(1).cpu().numpy() == hy[i:i + batch]).sum())
    for i in range(0, len(gz), batch):
        _, vv, _, s_ = net.forward_heads(torch.from_numpy(gx[i:i + batch]).to(device).float())
        vs.append(vv.cpu().numpy())
        if s_ is not None: ss.append(s_.cpu().numpy() * SCORE_SCALE)
    v = np.concatenate(vs); k = gz != 0
    out.update(human_policy_acc=hit / len(hy), gen_value_mse=float(((v[k] - gz[k]) ** 2).mean()))
    if ss: out['gen_score_mae'] = float(np.abs(np.concatenate(ss) - gsc).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selfplay', nargs='+', default=['runs/experiments/selfplay_cap_3300_20261019'])
    ap.add_argument('--init', default='checkpoints/shared_supervised_6x96_20261018/best.pt')
    ap.add_argument('--run', default='search_finetune_20261019')
    ap.add_argument('--value-fraction', type=float, default=1 / 20)
    ap.add_argument('--epochs', type=float, default=2.0)
    ap.add_argument('--batch', type=int, default=64, help='self-play rows per update')
    ap.add_argument('--human-batch', type=int, default=16)
    ap.add_argument('--gen-batch', type=int, default=16)
    ap.add_argument('--value-weight', type=float, default=1.0)
    ap.add_argument('--ownership-weight', type=float, default=0.5)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--seed', type=int, default=20261019)
    ap.add_argument('--gen-sources', choices=['gen', 'gen+gen2'], default='gen',
                    help="gen+gen2 = old AlphaGo-style data + the 1.5M RL-generated games (same split as train_value --gen-only)")
    ap.add_argument('--steps', type=int, default=None, help='override epochs-based step count')
    ap.add_argument('--search-mix', type=float, default=0.0,
                    help='value target = (1-m)*outcome + m*MCTS root value (self-play rows only)')
    ap.add_argument('--save-evals', action='store_true', help='keep a checkpoint at every evaluation')
    ap.add_argument('--dedup', action='store_true', help='drop exact duplicate games (identical move sequences)')
    ap.add_argument('--score-weight', type=float, default=0.0,
                    help='>0 adds a KataGo-style final-score head (Huber on margin/20), self-play + generated rows')
    args = ap.parse_args(); os.chdir(ROOT)
    out = ROOT / 'runs/experiments' / args.run; out.mkdir(parents=True, exist_ok=True)
    ckdir = ROOT / 'checkpoints' / args.run; ckdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); device = pick_device()
    if device.type != 'mps': raise RuntimeError('MPS unavailable')
    torch.mps.set_per_process_memory_fraction(.12)

    sp = SelfPlay([ROOT / d for d in args.selfplay], args.value_fraction, args.seed, search_mix=args.search_mix, dedup=args.dedup)
    memory_guard()
    human = GameData([ROOT / 'data/kgs.npz', ROOT / 'data/fox_dan.npz'])
    gen = ValueData([load_gen()] + ([load_gen('data/value/gen_rlcont/batch_*.npz', name='gen2')]
                                    if args.gen_sources == 'gen+gen2' else []))
    hx, hy, _ = human.sample(4096, np.random.default_rng(71), 'val', augment=False, threads=4)
    rows = gen.index['test'][np.random.default_rng(72).permutation(gen.size('test'))[:4096]]
    gen_eval = (*gen.batch_from_index(rows), gen.scores_from_index(rows))
    info = {'args': vars(args), 'games': sp.games, 'rejected_games': sp.rejected, 'duplicate_games': sp.duplicates,
            'train_positions': len(sp.train), 'val_positions': len(sp.val),
            'value_labelled_train_positions': int(sp.value_mask[sp.train].sum())}
    print(json.dumps(info), flush=True)

    ck = torch.load(ROOT / args.init, map_location='cpu', weights_only=False)
    cfg = NetConfig(**{**ck['config'], **({'score': True} if args.score_weight > 0 else {})})
    net = GoNet(cfg).to(device)
    missing, unexpected = net.load_state_dict(ck['model'], strict=False)
    if unexpected or any(not k.startswith('score_') for k in missing): raise ValueError(f'init mismatch {missing} {unexpected}')
    if missing: print('new head parameters:', missing, flush=True)
    torch.manual_seed(args.seed)
    base = evaluate(net, sp, (hx, hy), gen_eval, device)
    info['baseline'] = base; write_json(out / 'info.json', info); print('baseline', json.dumps(base), flush=True)

    # New score head learns 10x faster than the pretrained trunk/heads.
    groups = []
    for new in (False, True):
        for dec in (True, False):
            ps = [p for n_, p in net.named_parameters() if n_.startswith('score_') == new and (p.ndim > 1) == dec]
            if ps: groups.append({'params': ps, 'weight_decay': 1e-4 if dec else 0, 'mult': 10. if new else 1.})
    opt = torch.optim.SGD(groups, lr=args.lr, momentum=.9, nesterov=True)
    steps = args.steps or int(args.epochs * len(sp.train) / args.batch)
    rng = np.random.default_rng(args.seed); history = []; t0 = time.monotonic()
    training_mode(net)  # BN running stats frozen, all weights train
    for step in range(1, steps + 1):
        for g in opt.param_groups: g['lr'] = g['mult'] * args.lr * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / steps)))
        x, pi, z, own, vm, sc = sp.batch(sp.train[rng.integers(0, len(sp.train), args.batch)], rng)
        hxb, hyb, _ = human.sample(args.human_batch, rng, augment=True, threads=4)
        grows = gen.index['train'][rng.integers(0, gen.size('train'), args.gen_batch)]
        gxb, gzb, gob = gen.batch_from_index(grows, rng, augment=True); gsc = gen.scores_from_index(grows)
        xs = torch.from_numpy(np.concatenate([x, hxb, gxb])).to(device).float()
        pi_t, z_t, own_t, vm_t, hy_t, gz_t, go_t = to(device, pi, z, own, vm, hyb.astype(np.int64), gzb, gob)
        p, v, o, sh = net.forward_heads(xs)
        a, b = args.batch, args.batch + args.human_batch
        policy_loss = -(pi_t * F.log_softmax(p[:a], -1)).sum(-1).mean()
        human_loss = F.cross_entropy(p[a:b], hy_t)
        vmask = torch.cat([vm_t, gz_t != 0]); vtarget = torch.cat([z_t, gz_t]); vpred = torch.cat([v[:a], v[b:]])
        value_loss = F.mse_loss(vpred[vmask], vtarget[vmask]) if vmask.any() else v.sum() * 0
        own_loss = F.mse_loss(torch.cat([o[:a], o[b:]]), torch.cat([own_t, go_t]))
        loss = policy_loss + human_loss + args.value_weight * value_loss + args.ownership_weight * own_loss
        if sh is not None:
            st = torch.from_numpy(np.concatenate([sc, gsc]) / SCORE_SCALE).to(device).float()
            spred = torch.cat([sh[:a], sh[b:]]); ok = torch.isfinite(st)
            score_loss = F.huber_loss(spred[ok], st[ok], delta=1.0)
            loss = loss + args.score_weight * score_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True); opt.step()
        if step % args.eval_every == 0 or step == steps:
            rec = {'step': step, 'steps': steps, 'loss': loss.item(), 'seconds': time.monotonic() - t0,
                   **evaluate(net, sp, (hx, hy), gen_eval, device)}
            history.append(rec); training_mode(net); print(json.dumps(rec), flush=True)
            if args.save_evals:
                torch.save({'model': net.state_dict(), 'config': vars(cfg), 'step': step}, ckdir / f'step_{step:06d}.pt')
            write_json(out / 'history.json', history); memory_guard()
    torch.save({'model': net.state_dict(), 'config': vars(cfg), 'args': vars(args), 'history': history,
                'training': 'MCTS self-play fine-tune: visits policy + value + ownership (+ score)'}, ckdir / 'final.pt')
    print('saved', ckdir / 'final.pt', flush=True)


if __name__ == '__main__':
    main()
