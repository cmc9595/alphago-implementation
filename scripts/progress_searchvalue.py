"""Shared model with value target = 0.5*outcome + 0.5*MCTS value. Usage: progress_searchvalue.py [--watch]"""
import json, os, subprocess, sys, time
from datetime import datetime, timedelta
LOG = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'runs/round3_merged_train.log'
KEYS = [('selfplay_score_mae', '자가대국 집차이오차', '.1f'), ('gen_score_mae', '생성 집차이오차', '.1f'),
        ('selfplay_value_mse', '자가대국 가치MSE', '.4f'), ('gen_value_mse', '생성데이터 가치MSE', '.4f'),
        ('selfplay_ownership_mse', '소유권MSE', '.4f'), ('selfplay_policy_ce', '자가대국 정책오차', '.3f'),
        ('human_policy_acc', '사람수 정확도', '.1%')]

def show():
    print('3바퀴 학습 · 이 맥 9,293판 + M1 Max 20,000판 + 136판 (800회 탐색 정답) · 가치 전부 · 2바퀴 모델에서 출발')
    base, recs = None, []
    if os.path.exists(LOG):
        for l in open(LOG):
            if l.startswith('baseline '): base = json.loads(l[9:])
            elif l.startswith('{"step"'): recs.append(json.loads(l))
    total = recs[-1]['steps'] if recs else 15000; done = recs[-1]['step'] if recs else 0
    w = int(40 * done / total)
    print(f'[{"█" * w}{"░" * (40 - w)}] {done:,}/{total:,} 스텝 ({done / total:.1%})')
    if recs:
        per = recs[-1]['seconds'] / done; left = (total - done) * per
        print(f'{per:.2f}초/스텝 · 남은 시간 {left / 60:.0f}분 · 학습 종료 예상 {(datetime.now() + timedelta(seconds=left)).strftime("%H:%M")}')
    print(f'\n{"스텝":>6} ' + ' '.join(f'{n:>12}' for _, n, _ in KEYS))
    for label, r in ([('학습 전', base)] if base else []) + [(str(r['step']), r) for r in recs[-10:]]:
        print(f'{label:>6} ' + ' '.join(f'{format(r[k], f) if k in r else "-":>12}' for k, _, f in KEYS))
    print('\n가치MSE는 순수 최종 승패 기준(검증 판 단위 10%) · 학습 정답만 섞음')
    stage = open('runs/round3_merged_pipeline.log').read().strip().splitlines()[-1] if os.path.exists('runs/round3_merged_pipeline.log') else ''
    print('단계:', stage)
    import glob
    for m, lab in [('ab_r3m_vs_round2_400', 'vs 2바퀴 모델'), ('ab_r3m_vs_round1_400', 'vs 1바퀴 모델')]:
        g = [json.load(open(f)) for f in glob.glob(f'runs/matches/{m}/game*.json')]
        if g: print(f'비교 대국 ({lab}, 400회): {len(g)}/100판 · {sum(x["a_won"] for x in g)}승 ({sum(x["a_won"] for x in g)/len(g):.0%})')

while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(30)
