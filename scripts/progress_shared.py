"""Lightweight live dashboard for shared-model training; never starts training.

    .venv/bin/python scripts/progress_shared.py --watch
    .venv/bin/python scripts/progress_shared.py
"""
import argparse
from collections import deque
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = 'shared_supervised_6x96_20261018'
KST = ZoneInfo('Asia/Seoul')
LABELS = {'loading_data':'데이터 로딩', 'baseline_evaluation':'기준 모델 평가',
          'training':'공유 모델 학습', 'benchmark':'추론·MCTS 속도 비교',
          'matches':'동일 400회 대국 평가', 'completed':'완료',
          'failed':'실패', 'stopped':'중단', 'preparing':'준비',
          'waiting_for_other_gpu_job':'다른 GPU 작업 종료 대기'}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def records(path):
    try:
        with path.open() as stream:
            result = []
            for line in stream:
                try: result.append(json.loads(line))
                except ValueError: continue  # Ignore an incomplete final append.
            return result
    except OSError:
        return []


def duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def process_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False


def recent_seconds_per_step(recs):
    samples = deque(maxlen=10)
    previous = 0
    for rec in recs:
        step, rate = rec.get('step', 0), rec.get('updates_per_second', 0)
        delta, previous = step - previous, step
        # Head-only warmup is faster and should not predict full-trunk training.
        if delta > 0 and not rec.get('heads_only', False) and math.isfinite(rate) and rate > 0:
            samples.append((delta, delta/rate))
    return sum(seconds for _,seconds in samples)/sum(n for n,_ in samples) if samples else None


def spark(values):
    if not values: return ''
    values = values[-20:]
    lo, hi = min(values), max(values)
    symbols = '▁▂▃▄▅▆▇█'
    return ''.join(symbols[round((v-lo)/max(hi-lo,1e-12)*7)] for v in values)


def bar(done, total):
    filled = int(32*min(1,max(0,done/max(1,total))))
    return '█'*filled+'░'*(32-filled)


def render(run, now=None):
    now = now or datetime.now(KST)
    out = ROOT/'runs/experiments'/run
    state, pipeline = read_json(out/'state.json'), read_json(out/'pipeline.json')
    baseline = read_json(out/'baseline.json')
    manifest = read_json(out/'manifest.json')
    recs = records(ROOT/'checkpoints'/run/'metrics.jsonl')
    if not state and not pipeline:
        return f'{now:%m-%d %H:%M:%S} KST\n공유 모델 기록이 없습니다. run 이름을 확인해 주세요.'
    stage = pipeline.get('status',state.get('status','preparing'))
    lines = [f'{now:%m-%d %H:%M:%S} KST · 공유 정책·가치망',
             f'전체 단계: {LABELS.get(stage,stage)}  |  {run}']
    if pipeline.get('error'): lines.append('파이프라인 오류: '+pipeline['error'])
    if state.get('error'): lines.append('학습 오류: '+state['error'])
    pidfile = out/'runner.pid'
    elapsed = None
    if pidfile.exists():
        elapsed = max(0,now.timestamp()-pidfile.stat().st_mtime)
        lines.append('실행 경과 '+duration(elapsed)+' (로딩·검증·잠자기 포함)')
    done,total = state.get('step',0),state.get('steps',30000)
    lines += ['',f'학습 [{bar(done,total)}] {done:,}/{total:,} ({done/max(1,total):.1%})']
    batch = manifest.get('args',{}).get('batch_per_source',32)*2
    lines.append(f'단계: {"가치·소유권 head 적응" if state.get("heads_only") else "공통 trunk까지 공동 학습"} · 배치 {batch}')
    seconds_step = recent_seconds_per_step(recs)
    alive = process_alive(state.get('pid'))
    try:
        updated = datetime.strptime(state['updated'],'%Y-%m-%d %H:%M:%S').replace(tzinfo=KST)
        fresh = (now-updated).total_seconds()<180
    except (KeyError,ValueError):
        fresh = False
    active = stage=='training' and state.get('status')=='training' and alive and fresh
    if active and seconds_step is not None:
        remaining = max(0,total-done)*seconds_step
        eta = now+timedelta(seconds=remaining)
        lines.append(f'예상 남은 학습 {duration(remaining)} · 예상 학습 종료 {eta:%m-%d %H:%M:%S} KST')
        lines.append(f'최근 평균 {1/seconds_step:.2f} 스텝/초 · {batch/seconds_step:.0f} 국면/초')
        if elapsed is not None: lines.append('시작부터 예상 학습 완료까지 '+duration(elapsed+remaining))
        lines.append('※ 학습 연산 속도 기준. 남은 검증·후속 속도 측정·20판 평가는 별도입니다.')
    elif done>=total:
        lines.append('학습 완료 · 아래는 마지막 학습/검증 기록입니다.')
    elif state.get('status') in ('failed','stopped') or (state.get('status')=='training' and not alive):
        lines.append('학습 프로세스 중단 · 종료 시각 추정을 보류합니다.')
    elif not fresh and state.get('status')=='training':
        lines.append('진행 갱신 지연 · 잠자기/검증/일시정지 여부 확인 필요. ETA 보류.')
    else:
        lines.append('속도 측정 대기 · 본학습 기록이 쌓이면 ETA가 표시됩니다.')
    current = recs[-1] if recs else state
    if 'value_mse' in current:
        lines += ['',f'학습 수치 · {current.get("step",done):,}스텝 · 최근 100업데이트 평균',
                  f'  가치 MSE {current["value_mse"]:.4f} · 소유권 MSE {current["ownership_mse"]:.4f}',
                  f'  정책 CE {current["policy_ce"]:.4f} · 정책 정확도 {current["policy_acc"]:.2%}',
                  f'  클리핑 전 gradient norm {current["grad_norm_before_clip"]:.3f}']
    validations = [r for r in recs if 'validation' in r]
    if validations:
        last = validations[-1];v = last['validation'];all_value = v['value']['all']
        lines += ['',f'검증 수치 · {last["step"]:,}스텝에서 평가 · 동일 고정 표본',
                  f'  정책 정확도 {v["policy_acc"]:.2%} · 정책 CE {v["policy_ce"]:.4f}',
                  f'  가치 MSE {all_value["mse"]:.4f} · 승패 적중 {all_value["winner_acc"]:.2%}',
                  f'  소유권 MSE {v.get("ownership_mse",float("nan")):.4f}',
                  '  단계별 가치 MSE: '+' / '.join(f'{label} {v["value"][name]["mse"]:.3f}' for name,label in [('opening','포석'),('early','초반'),('middle','중반'),('late','후반')] if name in v['value'])]
        if baseline:
            bp,bv = baseline['policy_acc'],baseline['value']['all']['mse']
            lines.append(f'  기존 분리망 기준: 정책 {bp:.2%} · 가치 MSE {bv:.4f}')
            lines.append(f'  기준 대비: 정책 {(v["policy_acc"]-bp)*100:+.2f}%p · 가치 MSE {all_value["mse"]-bv:+.4f} (낮을수록 좋음)')
        lines.append('  검증 가치 MSE 추이 '+spark([r['validation']['value']['all']['mse'] for r in validations])+' (왼쪽→오른쪽, 검증 순서)')
        lines.append(f'  표본: 사람 {v["human_positions"]:,}국면 / 생성 {all_value["n"]:,}국면 · 과거 검증 분리 재사용')
    else:
        lines += ['','첫 검증 대기 (기본 1,000스텝마다 평가)']
    speed = read_json(out/'speed.json')
    if speed:
        lines += ['',f'속도 비교: 공유 모델 추론 {speed["inference_speedup"]:.2f}배 · MCTS {speed["mcts_speedup"]:.2f}배 (실측)']
    summary = read_json(ROOT/'runs/matches'/f'{run}_vs_separate_400'/'summary.json').get('summary',{})
    if summary:
        completed = summary['completed_games'];score = summary.get('A_win_rate')
        lines += ['',f'동일 400회 평가 [{bar(completed,20)}] {completed}/20판',
                  f'  공유망 {summary["A_wins"]}승 · 성적 {score:.1%} · 제외 {summary["capped_games"]}판' if score is not None else '  아직 유효한 대국 결과가 없습니다.']
        if stage=='matches' and completed and process_alive(pipeline.get('child_pid')):
            seconds = summary['minutes']*60/completed*max(0,20-completed)
            lines.append(f'  예상 남은 평가 {duration(seconds)} · 종료 {(now+timedelta(seconds=seconds)):%m-%d %H:%M:%S} KST')
    lines += ['',f'학습 기록 갱신: {state.get("updated","–")} KST',
              '화면 2초 간격 · 학습 수치는 100스텝, 검증 수치는 1,000스텝마다 갱신됩니다.',
              '이 화면은 기록만 읽습니다. 현재 바둑판 모델에는 자동 적용하지 않습니다.']
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default=DEFAULT_RUN)
    ap.add_argument('--watch',action='store_true')
    ap.add_argument('--interval',type=float,default=2.)
    args = ap.parse_args()
    if args.interval<.5: ap.error('interval must be at least 0.5 seconds')
    try:
        while True:
            if args.watch: print('\033[2J\033[H',end='')
            print(render(args.run),flush=True)
            if not args.watch: break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return


if __name__=='__main__':main()
