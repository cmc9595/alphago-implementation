"""Round-4 training comparison (A: round-4 only, B: + round-3 value). Usage: progress_round4.py [--watch]"""
import glob, json, os, sys, time
from datetime import datetime, timedelta
KEYS = [('selfplay_value_mse', '가치MSE', '.4f'), ('selfplay_policy_ce', '정책오차', '.3f'),
        ('selfplay_ownership_mse', '소유권MSE', '.4f'), ('gen_value_mse', '생성 가치MSE', '.4f'), ('human_policy_acc', '사람수', '.1%')]

def run(label, log):
    print(f'\n{label}')
    if not os.path.exists(log): print('  대기'); return
    base, info, recs = None, None, []
    for l in open(log):
        if l.startswith('baseline '): base = json.loads(l[9:])
        elif l.startswith('{"args'): info = json.loads(l)
        elif l.startswith('{"step"'): recs.append(json.loads(l))
    if info: print(f'  판 {info["games"]:,} · 정책 국면 {info.get("policy_train_positions", info["train_positions"]):,} · 가치 국면 {info["value_labelled_train_positions"]:,}')
    if not base: print('  데이터 읽는 중'); return
    if recs:
        r = recs[-1]; per = r['seconds'] / r['step']; left = (r['steps'] - r['step']) * per
        w = int(30 * r['step'] / r['steps'])
        print(f'  [{"█" * w}{"░" * (30 - w)}] {r["step"]:,}/{r["steps"]:,} · 남은 {left / 60:.0f}분 (종료 {(datetime.now() + timedelta(seconds=left)).strftime("%H:%M")})')
    print('  ' + f'{"":>8}' + ''.join(f'{n:>12}' for _, n, _ in KEYS))
    for lab, r in [('학습 전', base)] + [(str(r['step']), r) for r in recs[-6:]]:
        print('  ' + f'{lab:>8}' + ''.join(f'{format(r[k], f):>12}' for k, _, f in KEYS))

def match(name, label):
    g = [json.load(open(f)) for f in glob.glob(f'runs/matches/{name}/game*.json')]
    if g: print(f'  {label}: {len(g)}/100판 · A {sum(x["a_won"] for x in g)}승 ({sum(x["a_won"] for x in g) / len(g):.0%})')

def show():
    print('5바퀴 학습 · 3바퀴 모델에서 출발 · 1,600회 탐색 정답 · 가치는 학습 국면의 절반')
    run('5바퀴 (20,000판)', 'runs/round5_train.log')
    print('\n비교 대국 (400회 탐색)')
    match('ab_r5_vs_r3_400', '5바퀴 vs 3바퀴'); match('ab_r5_vs_r1_400', '5바퀴 vs 1바퀴')
    print('단계:', open('runs/round5_pipeline.log').read().strip().splitlines()[-1])

while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(30)
