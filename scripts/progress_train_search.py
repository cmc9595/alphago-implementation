"""Read-only dashboard for train_search.py; refresh with --watch."""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = 'round3_20k_seed20297031_20261102'


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def render(run):
    out = ROOT / 'runs/experiments' / run
    ckdir = ROOT / 'checkpoints' / run
    info = read_json(out / 'info.json', {})
    history = read_json(out / 'history.json', [])
    # The trainer prints dataset info before the potentially long baseline evaluation.
    log = ROOT / 'runs' / f'{run}_train.log'
    try:
        log_lines = log.read_text().splitlines()
    except OSError:
        log_lines = []
    for line in log_lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and 'train_positions' in rec:
            info = info or rec
    try:
        pid = int((ROOT / 'runs' / f'{run}_train.pid').read_text())
        os.kill(pid, 0)
        alive = True
    except (OSError, ValueError):
        alive = False
    done = (ckdir / 'final.pt').exists()
    status = '완료' if done else ('실행 중' if alive else '미실행 / 중단')
    lines = [f'{datetime.now():%m-%d %H:%M:%S} · MCTS 자가대국 모델 훈련',
             run, f'상태: {status}']
    if info:
        lines.append(f"대국 {info['games']:,}판 · 학습 {info['train_positions']:,} / 검증 {info['val_positions']:,}국면")
    if history:
        last = history[-1]
        step, total = last['step'], last['steps']
        fraction = min(1., step / max(1, total))
        filled = int(40 * fraction)
        lines.append(f"[{'█' * filled}{'░' * (40 - filled)}] {step:,}/{total:,}스텝 ({fraction:.1%})")
        previous = history[-2] if len(history) > 1 else {'step': 0, 'seconds': 0}
        rate = (last['seconds'] - previous['seconds']) / max(1, step - previous['step'])
        if alive and not done:
            remaining = max(0, total - step) * rate
            lines.append(f'남은 시간 약 {remaining / 3600:.1f}시간 · 종료 예상 {datetime.now() + timedelta(seconds=remaining):%m-%d %H:%M}')
        for key, label in [('loss', '학습 손실'), ('selfplay_policy_ce', '정책 CE'),
                           ('selfplay_value_mse', '가치 MSE'), ('selfplay_ownership_mse', '소유권 MSE'),
                           ('human_policy_acc', '사람 정책 정확도'), ('gen_value_mse', '생성데이터 가치 MSE')]:
            if key in last:
                lines.append(f'{label}: {last[key]:.4f}')
        lines.append('진행률·지표는 평가 시점마다 갱신됩니다.')
    elif alive:
        lines.append('기준 모델 평가 중' if info else '데이터 로딩 / 준비 중')
    if not alive and not done:
        missing = [p for p in ['data/kgs.npz', 'data/fox_dan.npz'] if not (ROOT / p).exists()]
        if not list((ROOT / 'data/value/gen').glob('batch_*.npz')):
            missing.append('data/value/gen/batch_*.npz')
        if missing:
            lines.append('필수 학습 데이터 없음: ' + ', '.join(missing))
        if log_lines:
            lines.extend(['최근 로그:', *log_lines[-5:]])
    lines.append(f'로그: {log.relative_to(ROOT)}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', nargs='?', default=DEFAULT_RUN)
    ap.add_argument('--watch', action='store_true')
    args = ap.parse_args()
    try:
        while True:
            if args.watch:
                print('\033[2J\033[H', end='')
            print(render(args.run), flush=True)
            if not args.watch:
                break
            time.sleep(15)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
