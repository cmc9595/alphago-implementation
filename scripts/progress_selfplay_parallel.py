"""Live view of a parallel self-play run. Usage: progress_selfplay_parallel.py [run] [--watch]"""
import glob, json, os, sys, time
from datetime import datetime, timedelta

RUN = next((a for a in sys.argv[1:] if not a.startswith('--')), 'selfplay_round2_3000_20261019')
D = f'runs/experiments/{RUN}'


def alive():
    try:
        os.kill(int(open(f'runs/{RUN}.pid').read()), 0); return True
    except Exception:
        return False


def show():
    total = json.load(open(f'{D}/manifest.json'))['args']['games'] if os.path.exists(f'{D}/manifest.json') else 0
    files = sorted(glob.glob(f'{D}/game_*.json'), key=os.path.getmtime)
    g = [json.load(open(f)) for f in files]
    n = len(g)
    print(f'MCTS 자가대국 · {RUN}')
    if os.path.exists(f'{D}/manifest.json'):
        print('모델:', json.load(open(f'{D}/manifest.json'))['args']['model'], '· 병렬 32판 · 트리 재사용 · 기권 없음')
    if not total:
        print('준비 중...'); return
    w = int(40 * n / total)
    print(f'[{"█" * w}{"░" * (40 - w)}] {n:,}/{total:,}판 ({n / total:.1%})')
    pos = sum(x['training_positions'] for x in g)
    print(f'학습 국면 {pos:,}개 (판당 {pos / max(n, 1):.0f}) · 유효 {sum(x["valid"] for x in g):,}판 · 무효 {sum(not x["valid"] for x in g)}판')
    if n >= 40:
        ts = [os.path.getmtime(f) for f in files[-200:]]
        per = (ts[-1] - ts[0]) / (len(ts) - 1)
        left = per * (total - n)
        print(f'속도 {per:.1f}초/판 (최근 {len(ts)}판) · 남은 시간 {left / 3600:.1f}시간 · 종료 예상 {(datetime.now() + timedelta(seconds=left)).strftime("%m-%d %H:%M")}')
    elif n:
        print('속도: 첫 32판이 동시에 끝나는 중이라 40판 이후 표시')
    status = '실행 중' if alive() else ('완료' if n >= total else '멈춤 (프로세스 없음)')
    print('상태:', status)


while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(15)
