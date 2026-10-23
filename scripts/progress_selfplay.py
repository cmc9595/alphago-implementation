"""Compact self-play dashboard, styled after progress.py / progress_value.py.

.venv/bin/python scripts/progress_selfplay.py --watch
Only reads logs; Ctrl+C stops the dashboard, not the generator.
"""
import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = 'selfplay_cap_3300_20261019'
KST = ZoneInfo('Asia/Seoul')


def read_json(path):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return {}


def alive(pid):
    if not isinstance(pid, int) or pid <= 0: return False
    try: os.kill(pid, 0); return True
    except PermissionError: return True
    except ProcessLookupError: return False


def duration(seconds):
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600); m, s = divmod(rem, 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


def bar(frac, width=40):
    n = int(max(0., min(1., frac))*width)
    return '█'*n+'░'*(width-n)


def spark(values, width=40):
    values = values[-width:]
    if not values: return '–'
    lo, hi = min(values), max(values)
    chars = '▁▂▃▄▅▆▇█'
    return ''.join(chars[int((v-lo)/(hi-lo+1e-12)*7)] for v in values)


class GameRecords:
    """Reuse parsed finished games; do not reread thousands of files each refresh."""
    def __init__(self): self.cache = {}
    def read(self, out):
        for path in out.glob('game_*.json'):
            if path not in self.cache:
                rec = read_json(path)
                if rec: self.cache[path] = rec
        return sorted(self.cache.values(), key=lambda r: r['game'])


def render(out, recs, now=None, process_alive=alive):
    now = now or datetime.now(KST)
    state = read_json(out/'state.json')
    if not state:
        return f'{now:%H:%M:%S} KST\nrun {out.name}  [기록 대기]'
    done, total = state.get('games_done', 0), state['games']
    status = state['status']
    try: age = (now-datetime.fromisoformat(state['updated'])).total_seconds()
    except (KeyError, ValueError): age = float('inf')
    terminal = status in ('completed','failed','interrupted')
    running = not terminal and process_alive(state.get('pid'))
    fresh = age < 180
    label = ('DONE' if status == 'completed' else 'FAILED' if status == 'failed' else
             'STOPPED' if not running else 'STALE' if not fresh else 'RUNNING')
    recent = [r for r in recs if r['game'] < done][-20:]
    avg = (sum(r['seconds'] for r in recent)/len(recent) if recent else
           state.get('total_seconds', 0)/done if done else None)
    lines = [f'{now:%H:%M:%S} KST', f'run {out.name}  [{label}]',
             f'{bar(done/max(1,total))} {done:,}/{total:,}판 ({done/max(1,total):.1%})']
    if avg:
        text = f'최근 {len(recent) or done}판 평균 {avg:.1f}초/판 · {3600/avg:.1f}판/시간'
        if running and fresh and status == 'selfplay':
            current = min(avg, max(0, state.get('current_game_seconds', 0))) if state.get('move') else 0
            remaining = max(0, (total-done)*avg-current)
            text += f'\n남은 약 {duration(remaining)}  (ETA {now+timedelta(seconds=remaining):%m-%d %H:%M} KST)'
        elif not terminal: text += '\n프로세스 중단 또는 갱신 지연 · ETA 보류'
        lines.append(text)
    else: lines.append('첫 판이 완료되면 생성 속도와 예상 종료 시각을 표시합니다.')
    measured = state.get('total_seconds', 0)
    if running: measured += state.get('current_game_seconds', 0)
    lines += ['', f'생성·판정 측정 누적 {duration(measured)}  (로딩·저장 시간 제외)',
              f'현재 {state.get("game_id", 0):,}번째 판 · {state.get("move", 0)}수' if running else f'생성 상태: {status}',
              f'유효 판정 {state.get("valid_games", 0):,}판 · 제외 {state.get("invalid_games", 0):,}판',
              f'정책 학습 후보 {state.get("training_positions", 0):,}국면 (400회 탐색분만)',
              f'가치용 1/20 선택 시 약 {state.get("training_positions", 0)/20:,.0f}국면 · 아직 선택/학습 안 함']
    if recs:
        full = sum(r['full_search_moves'] for r in recs)
        fast = sum(r['fast_search_moves'] for r in recs)
        lines += [f'400회 {full:,}수 / 100회 {fast:,}수 · 깊은 탐색 비율 {full/max(1,full+fast):.1%}',
                  f'판당 생성 시간 추이 {spark([r["seconds"] for r in recs])}  (최근 40판)',
                  f'최근 판: {recs[-1]["moves"]}수 · {recs[-1]["seconds"]:.1f}초 · GNU Go {recs[-1]["judge_seconds"]:.2f}초']
    lines += ['', '공유 정책·가치 모델 · 배치 32 · 25% 400회 / 75% 100회',
              '기보 생성만 진행 · 학습/모델 교체 없음',
              f'기록 갱신 {state["updated"]} · Ctrl+C: 이 화면만 종료']
    if state.get('error'): lines.append('오류: '+state['error'])
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=DEFAULT_RUN)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=float, default=5.)
    args = ap.parse_args()
    if args.interval < .5: ap.error('interval must be at least 0.5 seconds')
    out = ROOT/'runs/experiments'/args.run
    cache = GameRecords()
    tty = args.watch and sys.stdout.isatty()
    try:
        if tty: print('\033[?1049h\033[?25l',end='',flush=True)
        while True:
            text = render(out, cache.read(out))
            prefix = '\033[H\033[2J' if tty else ''
            print(prefix+text,flush=True)
            if not args.watch: break
            time.sleep(args.interval)
    except KeyboardInterrupt: pass
    finally:
        if tty: print('\033[?25h\033[?1049l',end='',flush=True)


if __name__ == '__main__': main()
