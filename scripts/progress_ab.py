"""Live view of the fine-tuned vs base A/B match. Usage: progress_ab.py [--watch]"""
import glob, json, math, sys, time
from datetime import datetime, timedelta
D = next(((a if a.startswith('runs/') else f'runs/matches/{a}') for a in sys.argv[1:] if not a.startswith('--')), 'runs/matches/ab_search_finetune_vs_shared_400'); TOTAL = int(next((a.split('=')[1] for a in sys.argv if a.startswith('--total=')), 50))

def show():
    g = [json.load(open(f)) for f in sorted(glob.glob(f'{D}/game*.json'))]
    n = len(g); w = sum(x['a_won'] for x in g)
    print('비교 대국:', D.split('/')[-1], '· 양쪽 400회 탐색 · A가 새 모델')
    print(f'[{"█"*int(30*n/TOTAL)}{"░"*(30-int(30*n/TOTAL))}] {n}/{TOTAL}판')
    if n:
        p = w/n; z = 1.96; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
        print(f'A {w}승 {n-w}패 · 승률 {p:.1%} · 95% 구간 {c-h:.1%}~{c+h:.1%}')
        print('A 흑일 때', sum(x['a_won'] for x in g if x['a_black']), '/', sum(x['a_black'] for x in g),
              '· A 백일 때', sum(x['a_won'] for x in g if not x['a_black']), '/', sum(not x['a_black'] for x in g))
        print('최근 20판', ''.join('●' if x['a_won'] else '○' for x in g[-20:]))
        verdict = '향상 확인 (구간 하한 > 50%)' if c-h > .5 else ('악화' if c+h < .5 else '아직 판단 불가')
        print('판정:', verdict)
        from os.path import getmtime
        ts = sorted(getmtime(f) for f in glob.glob(f'{D}/game*.json'))
        if len(ts) > 2 and n < TOTAL:
            per = (ts[-1]-ts[0])/(len(ts)-1)
            print('예상 종료', (datetime.now()+timedelta(seconds=per*(TOTAL-n))).strftime('%H:%M'))

while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(15)
