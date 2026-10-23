"""Korean progress viewer; standard library only, no GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent.parent
LABELS={'diagnostics':'대칭 평균 오차·흔들림·속도 평가','symmetry':'대칭 평균 vs 기본 (동일 시간)',
        'value':'새 가치망 vs SL 가치 head','prior':'RL 정책 vs SL 정책',
        'pachi5000':'Pachi 5,000회 탐색 상대','katago_rank1d':'KataGo 초단 모방 프로필 상대'}


def progress_bar(done,total,width=28):
    fraction=min(1.0,max(0.0,done/total)) if total else 0.0
    filled=int(width*fraction)
    return '█'*filled+'░'*(width-filled)+f' {fraction:.1%}'


def load_result(path):
    try:
        return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):
        return {}


def render(name):
    folder=ROOT/'runs'/'experiments'/name
    path=folder/'state.json'
    if not path.exists(): return '실험 시작 대기 중'
    try: state=json.loads(path.read_text())
    except json.JSONDecodeError: return '진행 기록을 읽는 중'
    args=state['args']
    total_games=2*(3*args['internal_pairs']+2*args['external_pairs'])
    completed=0
    for key,stage in state.get('stages',{}).items():
        if key!='diagnostics':
            completed+=load_result(Path(stage.get('artifact','/nonexistent'))).get('summary',{}).get('completed_games',0)
    done_stages=sum(s.get('status')=='done' for s in state.get('stages',{}).values())
    lines=[f"실험 {name}  [{state['status']}]",
           f"전체 대국  {progress_bar(completed,total_games,36)}  {completed}/{total_games}판",
           f"완료 단계 {done_stages}/{len(LABELS)} · 진행률은 완료한 판 기준 (시간 비율 아님)",'']
    for key,label in LABELS.items():
        stage=state.get('stages',{}).get(key,{})
        status=stage.get('status','대기')
        artifact=Path(stage.get('artifact','/nonexistent'))
        result={}
        if artifact.is_file():
            try: result=json.loads(artifact.read_text())
            except json.JSONDecodeError: pass
        summary=result.get('summary',{})
        if summary:
            target=2*state['args']['external_pairs' if key in ('pachi5000','katago_rank1d') else 'internal_pairs']
            n=summary['completed_games']; score=summary['A_win_rate']
            lines.append(f'{label}: [{status}]')
            lines.append(f'  {progress_bar(n,target)}  {n}/{target}판')
            if score is not None:
                lo,hi=summary['ci95']
                lines.append(f"  A 성적 {score:.1%} (Wilson 95% {lo:.1%}~{hi:.1%}) · 상한 종료 {summary['capped_games']}판 제외")
            a,b=summary.get('A_mean_move_seconds'),summary.get('B_mean_move_seconds')
            if a is not None and b is not None: lines.append(f'  실제 수당 연산 A {a:.3f}초 / B {b:.3f}초 (GPU 잠금 대기 제외)')
            if status=='running' and n:
                eta=summary['minutes']/n*(target-n)
                lines.append(f'  이 대국 실험 남은 시간 약 {eta:.0f}분 (잠자기·사용 부하에 따라 변동)')
        else:
            lines.append(f'{label}: [{status}]')
            if key=='diagnostics':
                total=args.get('positions',8192)
                n=total if status=='done' else 0
                lines.append(f'  {progress_bar(n,total)}  {n:,}/{total:,}국면')
            else:
                target=2*args['external_pairs' if key in ('pachi5000','katago_rank1d') else 'internal_pairs']
                lines.append(f'  {progress_bar(0,target)}  0/{target}판')
                if status=='running':lines.append('  첫 대국 진행 중 · 한 판이 끝나면 갱신됩니다.')
    if 'symmetry_decision' in state:
        d=state['symmetry_decision'];lines+=['',f"대칭 후보 {d['candidate']}개 · 현재 채택 {d['adopted']}개"]
    if 'selected_spec' in state: lines+=['','선택한 조합: '+state['selected_spec']]
    if 'error' in state: lines+=['','중단 원인: '+state['error']]
    lines+=['','진행 화면을 닫아도 실험은 계속됩니다. 현재 브라우저 대국은 변경하지 않습니다.']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='ensemble_20261017');ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    while True:
        if args.watch: print('\033[2J\033[H',end='')
        print(time.strftime('%H:%M:%S'),render(args.name),sep='\n',flush=True)
        if not args.watch:break
        time.sleep(30)

if __name__=='__main__':main()
