"""Read-only Korean dashboard for the three-cycle replay pilot."""
import argparse
import json
from pathlib import Path
import time

from progress_train_search import metrics_table

ROOT = Path(__file__).resolve().parent.parent
PHASES = {'preparing': '준비', 'selfplay': '자가대국 생성', 'pool_index': 'replay 풀 정리·캐시',
          'training': '후보 학습', 'fixed_evaluation': '고정셋 가치 평가', 't5_diagnostic': 'T5 참고 분석',
          'screen': '기준 모델 상대 100판', 'confirm': '독립 확인 대국 200판',
          'cycle_complete': '반복 완료', 'completed': '3회 시험 운영 완료', 'failed': '오류로 중단', 'stopped': '중단'}


def read(path, default):
    try: return json.loads(path.read_text())
    except (OSError, ValueError): return default


def render(run):
    out = ROOT / 'runs/replay' / run
    state = read(out / 'state.json', {})
    if not state: return '아직 실행 기록이 없습니다.'
    phase = state.get('phase', 'preparing')
    games_total = state.get('games_total', 2000)
    sampling = '전체 풀 대국 균등 추출' if state.get('sampling') == 'uniform_games' else '최신/과거 50:50'
    lines = [run, f"반복 {state.get('cycle', 0)}/3 · {PHASES.get(phase, phase)} · 갱신 {state.get('updated_at', '')}",
             f"탐색 {state.get('visits', 1200):,}/{state.get('fast_visits', 200):,}회 · 학습용 {state.get('full_prob', .06):.0%} · 반복당 {games_total:,}판 · {sampling} · 배치 {state.get('batch', 64):,} · {state.get('train_steps', 1000):,}스텝",
             f"기준 모델: {state.get('champion', '준비 중')}"]
    if state.get('learner'):
        lines.append('누적 학습 모델: ' + state['learner'])
    lines.append(f"가치 정답 사용: 추출 국면의 {state.get('value_fraction', .5):.0%} · 최종 승패 ±1")
    schedule = read(out / 'generation_schedule.json', {})
    for key in sorted(schedule, key=int):
        if int(key) > state.get('cycle', 0):
            override = schedule[key]
            lines.append(f"예정: {key}회차 탐색 {override.get('visits', 1600):,}/200회 · 학습용 {override.get('full_prob', .06):.0%}")
    if phase == 'selfplay':
        directory = ROOT / 'runs/experiments' / state['selfplay_run']
        files = list(directory.glob('game_*.json'))
        rows = [read(p, {}) for p in files]
        lines.append(f"생성 {len(files):,}/{games_total:,}판 · 유효 {sum(bool(r.get('valid')) for r in rows):,}판")
        lines.append('일반 자가대국 진행 스크립트에도 표시됩니다: ' + state['selfplay_run'])
    if phase == 'pool_index':
        lines.append(f"캐시·목록 확인 {state.get('indexed_games', 0):,}판")
    if 'pool_games' in state:
        lines.append(f"활성 훈련 풀 {state['pool_games']:,}판 · 최신 훈련 대국 {state.get('latest_games', 0):,}판")
    if state.get('train_dir'):
        train = ROOT / state['train_dir']
        info, history = read(train / 'info.json', {}), read(train / 'history.json', [])
        if history:
            last = history[-1]
            lines.append(f"후보 학습 {last['step']:,}/{last['steps']:,}스텝")
            if 'total_updates' in last:
                lines.append(f"누적 replay 업데이트: {last['total_updates']:,}스텝")
        lines.extend(metrics_table(info, history, rows=10))
        if history:
            lines.append('손실은 학습 배치, CE·MSE는 고정 분할된 내부 모니터링 국면 기준입니다.')
    if phase in ('screen', 'confirm'):
        match = ROOT / 'runs/matches' / state['match']
        summary = read(match / 'summary.json', {}).get('summary', {})
        lines.append(f"대국 {summary.get('completed_games', 0)}/{state['match_total']}판 · 후보 승률 {summary.get('A_win_rate', 0):.1%}")
    lines.append('반복 | 100판 선별 | 200판 확인 | 승격')
    for cycle in range(1, 4):
        result = read(out / f'cycle_{cycle:02d}/complete.json', {})
        if not result: continue
        screen, confirm = result['screen'], result.get('confirmation')
        score = f"{confirm['win_rate']:.1%}" if confirm else '미실시'
        lines.append(f"{cycle:>4} | {screen['win_rate']:.1%} | {score:>10} | {'예' if result['promoted'] else '아니오'}")
    if state.get('error'): lines.append('오류: ' + state['error'])
    if state.get('child_log'): lines.append('현재/최근 로그: ' + state['child_log'])
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run', nargs='?', default='replay_continuous_1600v_p06_5000_20261121')
    ap.add_argument('--watch', action='store_true')
    args = ap.parse_args()
    try:
        while True:
            if args.watch: print('\033[2J\033[H', end='')
            print(render(args.run), flush=True)
            if not args.watch: break
            time.sleep(15)
    except KeyboardInterrupt: pass


if __name__ == '__main__':
    main()
