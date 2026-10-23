"""One reproducible time-budget match: A selected policy+new value, B shared SL policy/value.

Run/resume: python scripts/run_candidate_match.py
Watch:      python scripts/run_candidate_match.py --progress --watch
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT=Path(__file__).resolve().parent.parent


def atomic(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)); tmp.replace(path)


def progress(folder):
    path=folder/'state.json'
    if not path.exists(): print('실험 준비 전'); return
    state=json.loads(path.read_text()); args=state['args']; target=2*args['pairs']
    artifact=Path(state['artifact']); summary={}
    if artifact.exists(): summary=json.loads(artifact.read_text()).get('summary',{})
    n=summary.get('completed_games',0); fill=int(32*min(n/target,1))
    print(datetime.now().strftime('%H:%M:%S'),state['status'])
    print('후보 엔진 비교: '+'█'*fill+'░'*(32-fill)+f' {n}/{target}판 ({n/target:.1%})')
    policy_label=args.get("policy","rl").upper()
    print(f"A: {policy_label} 정책 + 새 가치망 / B: SL 정책 + SL 가치 · 양쪽 수당 {args['seconds']}초 · 단일 방향")
    if summary:
        print('A 승률:',summary['A_win_rate'],'Wilson 95%:',summary['ci95'],
              '포석 쌍 기준 95%:',summary.get('paired_bootstrap_ci95'))
        print('수당 평균 탐색 A/B:',summary.get('A_mean_visits'),summary.get('B_mean_visits'))
        print('수당 평균 시간 A/B:',summary.get('A_mean_move_seconds'),summary.get('B_mean_move_seconds'))
        print('수순 상한으로 무효 처리:',summary.get('capped_games',0))
        if state['status']=='running' and n:
            print(f"남은 시간 참고: 약 {summary['minutes']/n*(target-n)/60:.1f}시간 (절전·부하에 따라 변동)")
    if state.get('error'): print('중단 원인:',state['error'])
    print('Ctrl+C는 진행 화면만 종료합니다.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='sl_value_2s_20261018')
    ap.add_argument('--policy',choices=['sl','rl'],default='sl')
    ap.add_argument('--pairs',type=int,default=50)
    ap.add_argument('--seconds',type=float,default=2.)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--progress',action='store_true')
    ap.add_argument('--watch',action='store_true')
    args=ap.parse_args()
    if args.pairs<1 or args.seconds<=0: ap.error('pairs and seconds must be positive')
    os.chdir(ROOT); folder=ROOT/'runs'/'experiments'/args.name
    if args.progress:
        while True:
            if args.watch: print('\033[2J\033[H',end='')
            progress(folder)
            if not args.watch: return
            time.sleep(30)
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'runner.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    modeldir=folder/'models'; modeldir.mkdir(exist_ok=True)
    models={}
    for key in ('sl','rl','value'):
        source=ROOT/'runs/experiments/fixed_visits_20261017/models'/(key+'.pt')
        dest=modeldir/(key+'.pt')
        if not dest.exists(): shutil.copy2(source,dest)
        models[key]=str(dest.relative_to(ROOT))
    def spec(policy,value):
        return f"mcts:{models[policy]}:20000:{models[value]}:sym=1:sec={args.seconds}"
    a,b=spec(args.policy,'value'),spec('sl','sl')
    cmd=[sys.executable,'scripts/match.py',a,b,'--pairs',str(args.pairs),'--parallel','2',
         '--opening-net',models['sl'],'--opening-moves','6','--seed','2026101801',
         '--max-moves','500','--resume','--name',args.name]
    parameters={'name':args.name,'pairs':args.pairs,'seconds':args.seconds,'policy':args.policy}
    sourcepaths=('alphago/mcts.py','scripts/match.py','scripts/run_candidate_match.py')
    manifest={'args':parameters,'A':a,'B':b,'command':cmd,
              'model_sha256':{k:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for k,p in models.items()},
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sourcepaths},
              'terminal_evaluation':'Stop at two passes; freeze learned value. Actual final score: GNU Go Chinese rules.',
              'note':'New code after terminal-stop repair. Previous matches are exploratory, not pooled into this result.'}
    planpath=folder/'plan.json'
    if planpath.exists() and json.loads(planpath.read_text())!=manifest:
        raise ValueError('Plan/models/code changed; choose a new name.')
    atomic(planpath,manifest)
    sourcedir=folder/'source'; sourcedir.mkdir(exist_ok=True)
    for p in sourcepaths: shutil.copy2(ROOT/p,sourcedir/Path(p).name)
    statepath=folder/'state.json'
    state=json.loads(statepath.read_text()) if statepath.exists() else {}
    if state.get('status')=='done': progress(folder); return
    state.update(args=parameters,status='prepared' if args.prepare else 'running',pid=os.getpid(),
                 artifact=str(ROOT/'runs'/'matches'/args.name/'summary.json'),
                 started=state.get('started',datetime.now().isoformat()))
    state.pop('error',None); atomic(statepath,state)
    if args.prepare: progress(folder); return
    env=dict(os.environ,PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.3',PYTORCH_MPS_LOW_WATERMARK_RATIO='0.25',PYTHONUNBUFFERED='1')
    try:
        with (folder/'match.log').open('a') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state.update(status='done',finished=datetime.now().isoformat()); atomic(statepath,state)
        progress(folder)
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}'); atomic(statepath,state); raise


if __name__=='__main__': main()
