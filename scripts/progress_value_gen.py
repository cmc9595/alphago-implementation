"""Value net trained only on RL-generated data (AlphaGo style). Usage: progress_value_gen.py [--watch]"""
import json, os, subprocess, sys, time
from datetime import datetime, timedelta
RUN, TOTAL = 'value_gen_only', 30000
BASE = {'gen_test': 0.464}  # value_own on the old generated test set (different test split; rough reference)


def spark(v, w=40):
    if not v: return ''
    v = v[-w:]; lo, hi = min(v), max(v); bars = '▁▂▃▄▅▆▇█'
    return ''.join(bars[int((x - lo) / max(hi - lo, 1e-9) * 7)] for x in v)


def show():
    p = f'checkpoints/{RUN}/metrics.jsonl'
    print('가치망 학습 · RL 생성 데이터만 (기존 24.8만 판 + 새 148.6만 판) · 6×96 별도 가치망')
    if not os.path.exists(p):
        print('시작 중...'); return
    L = [json.loads(l) for l in open(p)]
    r = L[-1]; done = r['step']; w = int(40 * done / TOTAL)
    print(f'[{"█" * w}{"░" * (40 - w)}] {done:,}/{TOTAL:,} 스텝 ({done / TOTAL:.1%}) · lr {r["lr"]}')
    if done < TOTAL:
        left = (TOTAL - done) * 256 / max(r['samples_per_s'], 1)
        print(f'{r["samples_per_s"]} 샘플/초 · 남은 시간 {left / 60:.0f}분 · 종료 예상 {(datetime.now() + timedelta(seconds=left)).strftime("%H:%M")}')
    print(f'학습 가치 MSE {r["train_value_mse"]:.4f} · 소유권 MSE {r.get("train_ownership_mse", 0):.4f}')
    print('학습 MSE 추이', spark([x['train_value_mse'] for x in L]))
    T = [x for x in L if 'test' in x]
    if T:
        print('\n 스텝   학습MSE  테스트전체  기존생성(gen)  새생성(gen2)  차이(테스트-학습)')
        for t in T[-8:]:
            te = t['test']
            g, g2 = te.get('gen', {}).get('mse'), te.get('gen2', {}).get('mse')
            print(f'{t["step"]:>6}   {t["train_value_mse"]:.4f}    {te["all"]["mse"]:.4f}      {g:.4f}        {g2:.4f}       {te["all"]["mse"] - t["train_value_mse"]:+.4f}')
        last = T[-1]['test']
        print('\n대국 단계별 MSE:', ' · '.join(f'{k} {last[k]["mse"]:.3f}' for k in ('opening', 'early', 'middle', 'late') if k in last))
        if 'ownership_point_acc' in last: print(f'소유권 점별 정확도 {last["ownership_point_acc"]:.1%}')
        print(f'참고: 기존 가치망(value_own) 기존 생성 데이터 MSE {BASE["gen_test"]} (테스트 분할이 달라 대략 비교) · AlphaGo 0.234')
    running = subprocess.run(['pgrep', '-f', f'train_value.py.*{RUN}'], capture_output=True).returncode == 0
    print('상태:', '실행 중' if running else ('완료' if done >= TOTAL else '멈춤'))


while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(30)
