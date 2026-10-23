"""Fixed-visit controls, queued after the existing time-budget experiment.

Progress: python scripts/run_fixed_visit_experiments.py --progress
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

ROOT = Path(__file__).resolve().parent.parent


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def specifications(models):
    def engine(policy, value, visits=400, symmetry=1):
        # Deliberately omit sec= so every move completes its fixed visit budget.
        return f'mcts:{models[policy]}:{visits}:{models[value]}:sym={symmetry}'
    return [
        ('visits_400_100', '동일 SL 엔진: A 400회 / B 100회', engine('sl','sl',400), engine('sl','sl',100)),
        ('visits_1600_400', '동일 SL 엔진: A 1600회 / B 400회', engine('sl','sl',1600), engine('sl','sl',400)),
        ('prior_400', '양쪽 400회: A RL 정책 / B SL 정책, 가치망은 SL로 동일', engine('rl','sl'), engine('sl','sl')),
        ('value_400', '양쪽 400회: A 새 가치망 / B SL 가치, 정책은 SL로 동일', engine('sl','value'), engine('sl','sl')),
        ('symmetry_400', '양쪽 400회: A 8방향 / B 1방향, SL 정책과 새 가치망 동일', engine('sl','value',symmetry=8), engine('sl','value')),
    ]


def progress(folder):
    if not (folder/'state.json').exists():
        print('아직 준비되지 않았습니다.'); return
    state=read(folder/'state.json'); total=state['args']['pairs']*2; completed=0
    print(datetime.now().strftime('%H:%M:%S'), state['status'])
    if state.get('waiting_for'): print('기존 실험 완료 대기:', state['waiting_for'])
    for stage in state['plan']:
        key,label,_,_=stage
        artifact=ROOT/'runs'/'matches'/(folder.name+'_'+key)/'summary.json'
        summary=read(artifact).get('summary',{}) if artifact.exists() else {}
        n=summary.get('completed_games',0); completed+=n
        bar='█'*int(24*n/total)+'░'*(24-int(24*n/total))
        print(f'{label}\n  {bar} {n}/{total}판')
        if summary:
            print('  A 승률:',summary.get('A_win_rate'),'95% 구간:',summary.get('ci95'))
            print('  수당 탐색 A/B:',summary.get('A_mean_visits'),summary.get('B_mean_visits'),
                  '수당 시간 A/B:',summary.get('A_mean_move_seconds'),summary.get('B_mean_move_seconds'))
    print(f'전체 {completed}/{5*total}판')
    if state.get('error'): print(state['error'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='fixed_visits_20261017')
    ap.add_argument('--after',default='ensemble_20261017')
    ap.add_argument('--pairs',type=int,default=10)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--progress',action='store_true')
    ap.add_argument('--watch',action='store_true')
    args=ap.parse_args()
    if args.pairs<1: ap.error('pairs must be positive')
    os.chdir(ROOT); folder=ROOT/'runs'/'experiments'/args.name
    if args.progress:
        while True:
            if args.watch: print('\033[2J\033[H',end='')
            progress(folder)
            if not args.watch: return
            time.sleep(30)
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'runner.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statepath=folder/'state.json'
    prior=read(statepath) if statepath.exists() else {}
    modeldir=folder/'models'; modeldir.mkdir(exist_ok=True)
    models={}
    for key in ('sl','rl','value'):
        source=ROOT/'runs'/'experiments'/args.after/'models'/(key+'.pt')
        dest=modeldir/(key+'.pt')
        if not dest.exists(): shutil.copy2(source,dest)
        models[key]=str(dest.relative_to(ROOT))
    plan=specifications(models)
    manifest={'args':{'name':args.name,'after':args.after,'pairs':args.pairs},'plan':plan,
              'model_sha256':{k:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for k,p in models.items()},
              'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ('scripts/match.py','alphago/mcts.py')}}
    manifest=json.loads(json.dumps(manifest))
    manifestpath=folder/'plan.json'
    if manifestpath.exists() and read(manifestpath)!=manifest:
        raise ValueError('Plan/models/code changed: use a new experiment name.')
    atomic(manifestpath,manifest)
    if prior.get('status')=='done': progress(folder); return
    state={**prior,'args':manifest['args'],'plan':manifest['plan'],'stages':prior.get('stages',{}),
           'status':'prepared' if args.prepare else 'waiting','pid':os.getpid()}
    atomic(statepath,state)
    if args.prepare: progress(folder); return
    try:
        dependency=ROOT/'runs'/'experiments'/args.after/'state.json'
        while True:
            parent=read(dependency)
            if parent['status']=='done': break
            if parent['status']!='running':
                raise RuntimeError('Preceding experiment did not finish successfully: '+parent['status'])
            os.kill(parent['pid'],0)
            state['waiting_for']=args.after; atomic(statepath,state)
            time.sleep(30)
        state.pop('waiting_for',None); state['status']='running'; atomic(statepath,state)
        env=dict(os.environ,PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.3',PYTORCH_MPS_LOW_WATERMARK_RATIO='0.25',
                 PYTHONUNBUFFERED='1',DATA_DIR=str(ROOT/'data'/'pachi'))
        for i,(key,label,a,b) in enumerate(plan):
            artifact=ROOT/'runs'/'matches'/(args.name+'_'+key)/'summary.json'
            if state['stages'].get(key,{}).get('status')=='done' and artifact.exists(): continue
            # Stop instead of mixing newly edited search code into a queued experiment.
            for p,digest in manifest['source_sha256'].items():
                if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Search/match code changed while queued: '+p)
            cmd=[sys.executable,'scripts/match.py',a,b,'--pairs',str(args.pairs),'--parallel','2',
                 '--opening-net',models['sl'],'--seed',str(61100+i*1000),'--resume','--name',args.name+'_'+key]
            state['current_stage']=key
            state['stages'][key]={'status':'running','started':datetime.now().isoformat(),'command':cmd,'artifact':str(artifact)}
            atomic(statepath,state); print('START',key,flush=True)
            with (folder/(key+'.log')).open('a') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            summary=read(artifact)['summary']
            state['stages'][key].update(status='done',finished=datetime.now().isoformat(),summary=summary)
            atomic(statepath,state); print('DONE',key,flush=True)
        state.update(status='done',finished=datetime.now().isoformat()); atomic(statepath,state)
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}'); atomic(statepath,state); raise


if __name__=='__main__': main()
