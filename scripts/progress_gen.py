"""Live view of AlphaGo-style value data generation. Usage: progress_gen.py [--watch]"""
import json, os, subprocess, sys, time
from datetime import datetime, timedelta
LOG, GAMES, BATCH = 'runs/gen_rlcont.log', 1_500_000, 256

def show():
    L = [x for x in (json.loads(l) for l in open(LOG) if l.startswith('{"batch')) if 'sec_total' in x] if os.path.exists(LOG) else []
    total = -(-GAMES // BATCH); n = len(L); kept = sum(x['games_kept'] for x in L)
    print('가치망 데이터 생성 (AlphaGo 방식) · SL 포석 → U번째 무작위 수 → RL 최종끼리 끝까지 → GNU Go 판정')
    print('끝까지 두는 RL: checkpoints/rl_6x96/final.pt · 판당 2국면 (U번째 수 직후 + 이후 무작위 1개)')
    w = int(40 * n / total)
    print(f'[{"█" * w}{"░" * (40 - w)}] {n:,}/{total:,}묶음 ({n / total:.1%})')
    print(f'판 {kept:,} / {GAMES:,} · 국면 약 {2 * kept:,}개 (U 이전에 끝난 판 제외)')
    if n >= 20:
        rec = L[-200:]; s = sum(x['sec_total'] for x in rec) / len(rec); left = (total - n) * s
        print(f'속도 묶음당 {s:.1f}초 (시간당 약 {3600 / s * 256:,.0f}판) · 남은 시간 {left / 3600:.1f}시간 · '
              f'종료 예상 {(datetime.now() + timedelta(seconds=left)).strftime("%m-%d %H:%M")}')
        recent = [x['sec_total'] for x in L[-40:]]
        lo, hi = min(recent), max(recent); bars = '▁▂▃▄▅▆▇█'
        print('최근 40묶음 시간', ''.join(bars[int((v - lo) / max(hi - lo, 1e-9) * 7)] for v in recent), f'({lo:.0f}~{hi:.0f}초)')
    running = subprocess.run(['pgrep', '-f', 'gen_value_data.py.*gen_rlcont'], capture_output=True).returncode == 0
    print('상태:', '실행 중' if running else ('완료' if n >= total else '멈춤 (프로세스 없음)'))

while True:
    if '--watch' in sys.argv: print('\033[2J\033[H', end='')
    show()
    if '--watch' not in sys.argv: break
    time.sleep(30)
