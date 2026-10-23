"""Policy-vs-policy comparison results. Usage: progress_policy_compare.py [--watch]"""
import json, sys, time
LOG = 'runs/policy_compare_20261020/results.log'
NAME = {'rl_6x96/final.pt': 'RL 최종', 'search_finetune_20261019': '1바퀴', 'search_round2_20261020': '2바퀴',
        'shared_supervised_6x96_20261018': '원래 공유'}
def short(p): return next((v for k, v in NAME.items() if k in p), p)
def show():
    print('정책 대 정책 비교 (탐색 없음, 400판씩, GNU Go 판정)')
    temp = '1.0'
    for line in open(LOG):
        if line.startswith('=='):
            temp = '0.3' if 'T0.3' in line else '1.0'; continue
        r = json.loads(line)
        lo, hi = r['ci95']
        print(f"온도 {temp} · {short(r['A']):>5} vs {short(r['B']):<5} · A 승률 {r['A_win_rate']:.1%} (95% {lo:.1%}~{hi:.1%})")
while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(15)
