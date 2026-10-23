"""KataGo human-profile rating ladder. Usage: progress_rating.py [--watch]"""
import glob, json, math, os, sys, time
ENG = [('sep_valueown800', 'SL 정책 + 기존 가치망 + MCTS 800'), ('shared_round1_800', '1바퀴 공유모델 + MCTS 800')]
RANKS = [('5k', '5급'), ('1k', '1급'), ('1d', '1단'), ('3d', '3단')]
def wilson(w, n, z=1.96):
    p = w / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d; return c-h, c+h
def show():
    print('기력 측정 · 상대: KataGo 사람 모방 모델(급수 프로필, 수당 40회) · 프로필당 40판')
    for key, label in ENG:
        print(f'\n{label}')
        for r, rl in RANKS:
            g = [json.load(open(f)) for f in glob.glob(f'runs/matches/rating_{key}_rank_{r}/game*.json')]
            g = [x for x in g if not x['capped']]; n = len(g); w = sum(x['a_won'] for x in g)
            bar = '█' * int(20 * n / 40) + '░' * (20 - int(20 * n / 40))
            if n:
                lo, hi = wilson(w, n)
                print(f'  {rl:>3} [{bar}] {n:>2}/40  우리 {w}승 {n-w}패 ({w/n:.0%}, 95% {lo:.0%}~{hi:.0%})')
            else:
                print(f'  {rl:>3} [{bar}]  0/40  대기')
    print('\n승률이 50% 근처가 되는 프로필이 대략 우리 수준 (KataGo가 흉내 낸 사람 기준, 서버 급수와 같지 않음)')
while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(30)
