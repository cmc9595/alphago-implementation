"""Controlled value-data experiment; run sequentially, progress with --progress.

Keeps the original game-held-out split. Splits its games into selection/report
halves; both halves have been used by historical experiments, so neither is a
new untouched test set. The training arms share initialization and all settings.
"""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.10')
import argparse
from collections import deque
from datetime import datetime, timedelta
import fcntl
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from alphago.gtp import load_net
from alphago.models import GoNet, NetConfig, pick_device
from alphago.value_data import ValueData, load_gen, load_rl, load_human

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'runs/experiments/value_source_ab_20261018'
PHASES = [(0, 30, 'opening'), (30, 80, 'early'), (80, 150, 'middle'), (150, 10000, 'late')]

def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def heldout_partition(rows, report=False):
    # Same game always goes into one half, including all its sampled plies.
    bucket = ((rows[:, 1] * 1103515245 + rows[:, 0] * 12345 + 20261018) % 2147483647) % 2
    return rows[bucket == int(report)]

def select_rows(rows, limit, seed):
    # Equal source/phase quotas where data exists, no duplication; retain raw
    # per-source and per-phase metrics as the primary interpretable quantities.
    rng = np.random.default_rng(seed)
    groups = []
    for source in np.unique(rows[:, 0]):
        for lo, hi, _ in PHASES:
            group = rows[(rows[:, 0] == source) & (rows[:, 2] >= lo) & (rows[:, 2] < hi)]
            if len(group):
                groups.append(group[rng.permutation(len(group))])
    quota = max(1, limit // max(1, len(groups)))
    return np.concatenate([g[:quota] for g in groups])

@torch.no_grad()
def evaluate(net, data, rows, device, batch):
    net.eval()
    values, targets = [], []
    for start in range(0, len(rows), batch):
        feats, z, _ = data.batch_from_index(rows[start:start + batch])
        _, v = net(torch.from_numpy(feats).to(device).float())
        values.append(v.cpu().numpy())
        targets.append(z)
    v, z = np.concatenate(values), np.concatenate(targets)
    masks = {'all': np.ones(len(rows), bool)}
    for i, source in enumerate(data.sources):
        masks[source.name] = rows[:, 0] == i
        masks[source.name + '_middle'] = (rows[:, 0] == i) & (rows[:, 2] >= 80) & (rows[:, 2] < 150)
    for lo, hi, name in PHASES:
        masks[name] = (rows[:, 2] >= lo) & (rows[:, 2] < hi)
    result = {}
    for name, mask in masks.items():
        mask = mask & (z != 0)
        if mask.any():
            p, target = (v[mask] + 1) / 2, (z[mask] + 1) / 2
            ece = 0.
            for i in range(10):
                b = (p >= i / 10) & (p < (i + 1) / 10 if i < 9 else p <= 1)
                if b.any():
                    ece += float(b.mean() * abs(p[b].mean() - target[b].mean()))
            result[name] = {'n': int(mask.sum()), 'mse': float(((v[mask] - z[mask]) ** 2).mean()),
                            'winner_acc': float((np.sign(v[mask]) == z[mask]).mean()), 'ece': ece}
    return result

def selection_score(metrics):
    # Fixed equal-source mean; no arm-specific validation population.
    return float(np.mean([metrics[s]['mse'] for s in ('gen', 'rl', 'human') if s in metrics]))

def lr_at(step, total, base):
    if step < 500:
        return base * (step + 1) / 500
    return base * .5 * (1 + np.cos(np.pi * (step - 500) / max(1, total - 500)))

def memory_guard():
    # macOS ru_maxrss is bytes. Stops this experiment only, never user apps.
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 8 * 1024**3:
        raise MemoryError('Experiment exceeded 8 GiB CPU peak RSS guard')
    text = subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True).stdout
    for line in text.splitlines():
        if 'free percentage:' in line:
            free = int(line.rsplit(':', 1)[1].strip().rstrip('%'))
            if free < 10:
                raise MemoryError(f'System memory free percentage {free}% < 10%; resume after freeing memory')

def audit(sources):
    result = {}
    for source in sources:
        ends = source.offsets[1:]
        lengths = np.diff(source.offsets)
        passed = ((lengths >= 2) & (source.moves[ends - 1] == 361) & (source.moves[ends - 2] == 361))
        result[source.name] = {'games': len(lengths), 'positions': sum(map(len, source.plies)),
                               'two_pass_end': int(passed.sum()), 'length_ge400': int((lengths >= 400).sum()),
                               'unknown_value_games': int((source.z_black == 0).sum())}
    return result

def duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def recent_step_seconds(path, effective_batch):
    """Use recent measured training throughput; validation/pauses are excluded."""
    recent = deque(maxlen=10)
    previous = 0
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
            step, rate = rec['step'], rec['samples_per_s']
            delta = step - previous
            previous = step
            if delta > 0 and np.isfinite(rate) and rate > 0:
                recent.append((delta, delta * effective_batch / rate))
        except (ValueError, KeyError, TypeError):
            continue  # A writer can still be appending the last line.
    if not recent:
        return None
    return sum(seconds for _, seconds in recent) / sum(steps for steps, _ in recent)

def remaining_estimates(state, speeds):
    """Sequential completion times; an unstarted arm borrows its predecessor's speed."""
    fallback = next((speeds[a] for a in ('generated', 'mixed') if speeds.get(a) is not None), None)
    elapsed, result = 0., {}
    for arm in ('mixed', 'generated'):
        remaining = max(0, state['steps'] - state.get(arm, {}).get('step', 0))
        speed = speeds.get(arm)
        if remaining and speed is None:
            speed = fallback
        seconds = remaining * speed if remaining and speed is not None else (0. if not remaining else None)
        elapsed = elapsed + seconds if elapsed is not None and seconds is not None else None
        result[arm] = {'remaining': seconds, 'finish_after': elapsed,
                       'borrowed_speed': remaining > 0 and speeds.get(arm) is None}
    return result, elapsed

def progress():
    path = OUT / 'state.json'
    if not path.exists():
        print('아직 시작하지 않았습니다.'); return
    state = json.loads(path.read_text())
    manifest_path = OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    batch = manifest.get('args', {}).get('effective_batch', 256)
    speeds = {arm: recent_step_seconds(ROOT / 'checkpoints' / f'value_ab_{arm}_20261018/metrics.jsonl', batch)
              for arm in ('mixed', 'generated')}
    estimates, total_remaining = remaining_estimates(state, speeds)
    now = datetime.now()
    updated = datetime.strptime(state['updated'], '%Y-%m-%d %H:%M:%S') if state.get('updated') else now
    active = state['status'].startswith('training_') and (now - updated).total_seconds() < 120
    pid_path = OUT / 'runner.pid'
    if pid_path.exists():
        try:
            os.kill(int(pid_path.read_text()), 0)
        except (ProcessLookupError, ValueError):
            active = False
        except PermissionError:
            pass
    print('가치망 데이터 구성 비교 ·', state['status'])
    elapsed = max(0, now.timestamp() - pid_path.stat().st_mtime) if pid_path.exists() else None
    if elapsed is not None:
        print('실행 경과:', duration(elapsed))
    for arm in ('mixed', 'generated'):
        d = state.get(arm, {})
        step, total = d.get('step', 0), state['steps']
        width = int(30 * min(1, step / total))
        print(f'{arm:10s} [{"█" * width}{"░" * (30-width)}] {step:,}/{total:,}')
        estimate = estimates[arm]
        if step >= total:
            print('  학습 완료')
        elif active and estimate['remaining'] is not None:
            finish = now + timedelta(seconds=estimate['finish_after'])
            prefix = '예상 학습 소요' if not step else '예상 남은 학습'
            borrowed = ' · 앞선 실험 속도로 추정' if estimate['borrowed_speed'] else ''
            print(f'  {prefix} {duration(estimate["remaining"])} · 예상 종료 {finish:%m-%d %H:%M:%S}{borrowed}')
        if 'train_value_mse' in d:
            print(f'  학습 MSE {d["train_value_mse"]:.4f} · {d["samples_per_s"]:.0f} 국면/초')
        if 'validation' in d:
            m = d['validation']
            print(f'  검증 전체 MSE {m["all"]["mse"]:.4f} · 중반 {m.get("middle", {}).get("mse", float("nan")):.4f} · 생성자료 {m["gen"]["mse"]:.4f}')
    if active and total_remaining is not None:
        finish = now + timedelta(seconds=total_remaining)
        print(f'전체 남은 학습 {duration(total_remaining)} · 예상 학습 종료 {finish:%m-%d %H:%M:%S}')
        if elapsed is not None:
            print('예상 총소요:', duration(elapsed + total_remaining))
        print('최근 10개 기록의 속도 기준 · 검증/최종 평가·잠자기 시간은 별도')
    elif state['status'] == 'completed':
        print('전체 실험 완료')
    elif state['status'] == 'report_evaluation':
        print('학습 완료 · 최종 평가 중 (종료 시각 추정 제외)')
    else:
        print('종료 시각 추정 대기: 학습 시작 전이거나 진행 갱신이 중단된 상태입니다.')
    print('갱신:', state.get('updated'), '·', state.get('error', ''))

def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    lock = open(OUT / 'experiment.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    memory_guard()
    state = {'status': 'loading_data', 'steps': args.steps, 'pid': os.getpid()}
    def status():
        state['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        write_json(OUT / 'state.json', state)
    status()
    try:
        sources = [load_gen(), load_rl(), load_human()]
        data = ValueData(sources)
        write_json(OUT / 'audit.json', audit(sources))
        validation = select_rows(heldout_partition(data.index['test']), args.eval_positions, 71)
        report = select_rows(heldout_partition(data.index['test'], True), args.eval_positions, 72)
        rowfile = OUT / 'evaluation_rows.npz'
        if rowfile.exists():
            with np.load(rowfile) as old:
                if not np.array_equal(old['validation'], validation) or not np.array_equal(old['report'], report):
                    raise ValueError('Data changed since experiment preparation')
        else:
            np.savez_compressed(rowfile, validation=validation, report=report)
        init_path = ROOT / 'checkpoints/rl_6x96/final.pt'
        init = torch.load(init_path, map_location='cpu', weights_only=False)
        cfg = NetConfig(**{**init['config'], 'policy': False, 'ownership': True})
        files = sorted((ROOT / 'data/value/gen').glob('batch_*.npz')) + sorted((ROOT / 'checkpoints/rl_6x96/games').glob('iter*.npz')) + [ROOT / 'data/value/human_kgs.npz']
        manifest = {'args': vars(args), 'initialization_sha256': digest(init_path), 'evaluation_rows_sha256': digest(rowfile),
                    'config': vars(cfg), 'train_positions': int(len(data.index['train'])),
                    'validation_positions': len(validation), 'report_positions': len(report),
                    'selection': 'Equal-source validation MSE; identical held-out rows for both arms',
                    'limitations': ['Historical validation games reused; not an untouched final test',
                                    'No original per-batch producer-model hashes; final-RL provenance inferred from generation script/logs',
                                    'Existing GNU Go labels reused, not guaranteed perfect',
                                    'Two arms use identical smaller microbatches with gradient accumulation; BatchNorm differs from historical batch256'],
                    'files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in files]}
        manifest_path = OUT / 'manifest.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('Experiment settings/files changed; use the original command to resume')
        else:
            write_json(manifest_path, manifest)
        print(json.dumps({'audit': audit(sources), 'validation': len(validation), 'report': len(report)}), flush=True)
        device = pick_device()
        if device.type != 'mps' and not args.allow_cpu:
            raise RuntimeError('MPS unavailable; refusing an accidental multi-day CPU training run')
        if device.type == 'mps':
            torch.mps.set_per_process_memory_fraction(.12)
        memory_guard()
        baseline_path = OUT / 'baseline_validation.json'
        if not baseline_path.exists():
            state['status'] = 'baseline_validation'; status()
            baseline = {}
            for label, path in [('sl', 'checkpoints/sl_6x96/final.pt'), ('value_own', 'checkpoints/value_own/best.pt')]:
                net = load_net(str(ROOT / path), device=device)
                baseline[label] = evaluate(net, data, validation, device, args.microbatch)
                del net
                if device.type == 'mps': torch.mps.empty_cache()
            write_json(baseline_path, baseline)
        full_train = data.index['train']
        for arm in ('mixed', 'generated'):
            run_dir = ROOT / 'checkpoints' / ('value_ab_' + arm + '_20261018')
            run_dir.mkdir(exist_ok=True)
            torch.manual_seed(args.seed)
            net = GoNet(cfg).to(device)
            net.load_state_dict({k: v for k, v in init['model'].items() if not k.startswith('policy_')}, strict=False)
            decay = [p for p in net.parameters() if p.ndim > 1]
            no_decay = [p for p in net.parameters() if p.ndim <= 1]
            opt = torch.optim.SGD([{'params': decay, 'weight_decay': 1e-4}, {'params': no_decay, 'weight_decay': 0}], lr=args.lr, momentum=.9, nesterov=True)
            step, best = 0, float('inf')
            rng = np.random.default_rng(args.seed)
            last = run_dir / 'last.pt'
            if last.exists():
                ck = torch.load(last, map_location='cpu', weights_only=False)
                net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
                step, best = ck['step'], ck['best_score']
                rng.bit_generator.state = ck['numpy_rng']
                torch.set_rng_state(ck['torch_rng'])
            data.index['train'] = full_train if arm == 'mixed' else full_train[full_train[:, 0] == 0]
            def save(path):
                tmp = path.with_suffix('.tmp')
                torch.save({'model': net.state_dict(), 'optimizer': opt.state_dict(), 'config': vars(cfg),
                            'step': step, 'best_score': best, 'numpy_rng': rng.bit_generator.state,
                            'torch_rng': torch.get_rng_state(), 'args': vars(args)}, tmp)
                tmp.replace(path)
            state['status'] = 'training_' + arm
            state[arm] = {'step': step}; status()
            net.train()
            started, sum_v, sum_o, nlog = time.monotonic(), 0., 0., 0
            while step < args.steps:
                for group in opt.param_groups: group['lr'] = lr_at(step, args.steps, args.lr)
                opt.zero_grad(set_to_none=True)
                for _ in range(args.effective_batch // args.microbatch):
                    feats, z, own = data.sample(args.microbatch, rng)
                    x = torch.from_numpy(feats).to(device).float()
                    target = torch.from_numpy(z).to(device)
                    _, v, o = net.forward_all(x)
                    known = target != 0
                    vl = F.mse_loss(v[known], target[known]) if known.any() else v.sum() * 0
                    ol = F.mse_loss(o, torch.from_numpy(own).to(device))
                    ((vl + 1.5 * ol) * args.microbatch / args.effective_batch).backward()
                    sum_v += float(vl.detach()) * args.microbatch / args.effective_batch
                    sum_o += float(ol.detach()) * args.microbatch / args.effective_batch
                    del x, target, v, o, vl, ol
                opt.step(); step += 1; nlog += 1
                if step % 100 == 0 or step == args.steps:
                    rec = {'step': step, 'train_value_mse': sum_v / nlog, 'train_ownership_mse': sum_o / nlog,
                           'samples_per_s': nlog * args.effective_batch / (time.monotonic() - started), 'lr': opt.param_groups[0]['lr']}
                    if step % args.eval_every == 0 or step == args.steps:
                        rec['validation'] = evaluate(net, data, validation, device, args.microbatch)
                        net.train()
                        score = selection_score(rec['validation'])
                        if score < best:
                            best = score; save(run_dir / 'best.pt')
                    save(last)
                    state[arm].update(rec); status()
                    with open(run_dir / 'metrics.jsonl', 'a') as f: f.write(json.dumps(rec) + '\n')
                    print(arm, json.dumps(rec), flush=True)
                    sum_v = sum_o = 0.; nlog = 0; started = time.monotonic()
                    memory_guard()
            save(run_dir / 'final.pt')
            del net, opt
            if device.type == 'mps': torch.mps.empty_cache()
        data.index['train'] = full_train
        state['status'] = 'report_evaluation'; status()
        results = {}
        models = [('sl', 'checkpoints/sl_6x96/final.pt'), ('value_own', 'checkpoints/value_own/best.pt')] + [(a, f'checkpoints/value_ab_{a}_20261018/best.pt') for a in ('mixed', 'generated')]
        for label, path in models:
            net = load_net(str(ROOT / path), device=device)
            results[label] = evaluate(net, data, report, device, args.microbatch)
            del net
            if device.type == 'mps': torch.mps.empty_cache()
        write_json(OUT / 'report_results.json', results)
        state['status'] = 'completed'; status()
    except BaseException as exc:
        state['status'] = 'stopped' if isinstance(exc, KeyboardInterrupt) else 'failed'
        state['error'] = repr(exc); status()
        raise

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--progress', action='store_true')
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--microbatch', type=int, default=64)
    ap.add_argument('--effective-batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=.02)
    ap.add_argument('--eval-every', type=int, default=2000)
    ap.add_argument('--eval-positions', type=int, default=12000)
    ap.add_argument('--seed', type=int, default=20261018)
    ap.add_argument('--allow-cpu', action='store_true')
    args = ap.parse_args()
    os.chdir(ROOT)
    if args.progress or args.watch:
        while True:
            if args.watch: print('\033[2J\033[H', end='')
            progress()
            if not args.watch: break
            time.sleep(10)
    else:
        if args.effective_batch % args.microbatch: ap.error('effective batch must divide into microbatches')
        run(args)

if __name__ == '__main__': main()
