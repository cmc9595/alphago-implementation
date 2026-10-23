"""Value capacity test: 8x128 vs 6x96 (value_own). Usage: progress_value_capacity.py [--watch]"""
import json, os, sys, time
from datetime import datetime, timedelta
BASE = {'all': 0.6207, 'gen_test': 0.464}  # value_own 30k steps
RUNS = [('value_8x128', '8×128 처음부터'), ('value_own_cont', '6×96 이어서 30k')]
def show():
    print('가치망 용량 시험 · 같은 데이터·테스트셋 · 기준 value_own(6×96): 전체 0.621 / 생성 0.464')
    for run, label in RUNS:
        p = f'checkpoints/{run}/metrics.jsonl'
        if not os.path.exists(p):
            print(f'{label}: 대기'); continue
        L = [json.loads(l) for l in open(p)]
        r = L[-1]; done = r['step']; w = int(30 * done / 30000)
        print(f'\n{label}  [{"█"*w}{"░"*(30-w)}] {done:,}/30,000 · 학습 MSE {r["train_value_mse"]}')
        if done < 30000:
            left = (30000 - done) * 256 / max(r['samples_per_s'], 1)
            print(f'  {r["samples_per_s"]} 샘플/초 · 종료 예상 {(datetime.now()+timedelta(seconds=left)).strftime("%H:%M")}')
        for t in [x for x in L if 'test' in x][-4:]:
            a, g = t['test']['all']['mse'], t['test'].get('gen_test', {}).get('mse')
            print(f'  {t["step"]:>6} 테스트 전체 {a:.4f} ({a-BASE["all"]:+.4f}) · 생성 {g:.4f} ({g-BASE["gen_test"]:+.4f})')
while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(30)
