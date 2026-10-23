"""Shared model trained on self-play + human + RL-generated (gen+gen2) data. Usage: progress_shared_gen2.py [--watch]"""
import json, os, subprocess, sys, time
from datetime import datetime, timedelta
LOG = 'runs/shared_gen2_20261022.log'
KEYS = [('selfplay_policy_ce', '자가대국 정책오차', '.3f'), ('selfplay_value_mse', '자가대국 가치MSE', '.3f'),
        ('gen_value_mse', '생성데이터 가치MSE', '.3f'), ('selfplay_ownership_mse', '소유권MSE', '.3f'),
        ('human_policy_acc', '사람수 정확도', '.1%')]

def show():
    print('공유모델 학습 · 2바퀴 모델에서 출발 · 스텝당 생성 128(가치+소유권) + 사람 32(정책) + 자가대국 32(정책+소유권, 가치 1/20)')
    if not os.path.exists(LOG): print('시작 전'); return
    base, recs = None, []
    for l in open(LOG):
        if l.startswith('baseline '): base = json.loads(l[9:])
        elif l.startswith('{"step"'): recs.append(json.loads(l))
    total = recs[-1]['steps'] if recs else 30000; done = recs[-1]['step'] if recs else 0
    w = int(40 * done / total)
    print(f'[{"█" * w}{"░" * (40 - w)}] {done:,}/{total:,} 스텝 ({done / total:.1%})')
    if recs:
        per = recs[-1]['seconds'] / done; left = (total - done) * per
        print(f'{per:.2f}초/스텝 · 남은 시간 {left / 60:.0f}분 · 종료 예상 {(datetime.now() + timedelta(seconds=left)).strftime("%H:%M")}')
    print(f'\n{"스텝":>6} ' + ' '.join(f'{name:>12}' for _, name, _ in KEYS))
    rows = ([('학습 전', base)] if base else []) + [(str(r['step']), r) for r in recs[-8:]]
    for label, r in rows:
        print(f'{label:>6} ' + ' '.join(f'{format(r[k], f):>12}' for k, _, f in KEYS))
    print('\n생성데이터 가치MSE는 기존 생성(gen) 테스트 4,096국면 · 별도 가치망 기준: 기존 0.457 / 새 0.456 (전체 테스트셋)')
    running = subprocess.run(['pgrep', '-f', 'train_search.py.*shared_gen2'], capture_output=True).returncode == 0
    print('상태:', '실행 중' if running else ('완료' if done >= total else '멈춤'))

while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(30)
