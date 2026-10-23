"""Resumable, sequential experiments. No training and no modification of the live board."""
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


def atomic(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)


def strong_a(s):
    # Require both single-game Wilson and opening-pair bootstrap to favor A.
    return (s['games']>=16 and s['ci95'] and s['ci95'][0]>.5 and
            s.get('paired_bootstrap_ci95') and s['paired_bootstrap_ci95'][0]>.5)


def strong_b(s):
    return (s['games']>=16 and s['ci95'] and s['ci95'][1]<.5 and
            s.get('paired_bootstrap_ci95') and s['paired_bootstrap_ci95'][1]<.5)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='ensemble_20261017')
    ap.add_argument('--internal-pairs',type=int,default=10)
    ap.add_argument('--external-pairs',type=int,default=25)
    ap.add_argument('--seconds',type=float,default=.35)
    ap.add_argument('--positions',type=int,default=8192)
    args=ap.parse_args()
    os.chdir(ROOT)
    folder=ROOT/'runs'/'experiments'/args.name
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'runner.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This experiment is already running.')
    state_path=folder/'state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'stages':{}}
    # Also recognizes a runner started before file-lock support was added.
    old_pid=state.get('pid')
    if state.get('status')=='running' and old_pid and old_pid!=os.getpid():
        try:
            os.kill(old_pid,0)
        except ProcessLookupError:
            pass
        else:
            raise SystemExit(f'This experiment is already running (PID {old_pid}).')
    state.update(status='running',pid=os.getpid(),started=state.get('started',datetime.now().isoformat()),args=vars(args))
    modeldir=folder/'models';modeldir.mkdir(exist_ok=True)
    origins={'sl':'checkpoints/sl_6x96/final.pt','rl':'checkpoints/rl_6x96/final.pt','value':'checkpoints/value_own/best.pt'}
    models={}
    for key,original in origins.items():
        dest=modeldir/(key+'.pt')
        if not dest.exists(): shutil.copy2(ROOT/original,dest)
        models[key]=str(dest.relative_to(ROOT))
    state['model_sha256']={k:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for k,p in models.items()}
    old_manifest=folder/'plan.json'
    plan={'args':vars(args),'models':models,'hashes':state['model_sha256']}
    if old_manifest.exists() and json.loads(old_manifest.read_text())!=plan:
        raise ValueError('experiment plan changed; choose a new --name')
    atomic(old_manifest,plan)
    env=dict(os.environ,PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.3',PYTORCH_MPS_LOW_WATERMARK_RATIO='0.25',
             PYTHONUNBUFFERED='1',DATA_DIR=str(ROOT/'data'/'pachi'))

    def run(key,command,artifact):
        prev=state['stages'].get(key,{})
        if prev.get('status')=='done' and artifact.exists():
            return json.loads(artifact.read_text())
        state['current_stage']=key
        state['stages'][key]={'status':'running','started':datetime.now().isoformat(),'command':command,'artifact':str(artifact)}
        atomic(state_path,state)
        print('START',key,flush=True)
        with (folder/(key+'.log')).open('a') as log:
            subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads(artifact.read_text())
        state['stages'][key].update(status='done',finished=datetime.now().isoformat())
        if 'summary' in result: state['stages'][key]['summary']=result['summary']
        atomic(state_path,state)
        print('DONE',key,flush=True)
        return result

    def spec(policy,value,sym):
        return f'mcts:{policy}:20000:{value}:sym={sym}:sec={args.seconds}'

    def match(key,a,b,pairs,seed):
        name=args.name+'_'+key
        cmd=[sys.executable,'scripts/match.py',a,b,'--pairs',str(pairs),'--parallel','2',
             '--opening-net',models['sl'],'--seed',str(seed),'--resume','--name',name]
        return run(key,cmd,ROOT/'runs'/'matches'/name/'summary.json')['summary']

    try:
        diag=run('diagnostics',[sys.executable,'scripts/compare_ensembles.py','--positions',str(args.positions),
                                '--seconds',str(args.seconds),'--policy',models['sl'],'--value',models['value'],
                                '--out',str(folder/'diagnostics.json')],folder/'diagnostics.json')
        candidate=diag['candidate_symmetries']
        sym=1; value=models['value']; policy=models['sl']
        s=match('symmetry',spec(policy,value,candidate),spec(policy,value,1),args.internal_pairs,47100)
        if strong_a(s): sym=candidate
        state['symmetry_decision']={'candidate':candidate,'adopted':sym,'criterion':'Both 95% intervals favor candidate; otherwise keep incumbent.'}
        atomic(state_path,state)
        s=match('value',spec(policy,value,sym),spec(policy,models['sl'],sym),args.internal_pairs,48100)
        if strong_b(s): value=models['sl']
        s=match('prior',spec(models['rl'],value,sym),spec(models['sl'],value,sym),args.internal_pairs,49100)
        if strong_a(s): policy=models['rl']
        ours=spec(policy,value,sym)
        state['selected_spec']=ours
        atomic(folder/'recommendation.json',{'spec':ours,'symmetries':sym,'policy':policy,'value':value,
               'note':'Conservative exploratory selection. Live board unchanged. Grade not inferred from a human imitation profile.'})
        atomic(state_path,state)
        pachi='gtp:data/pachi/pachi -d 0 -s {seed} -t =5000 threads=1,max_tree_size=128'
        match('pachi5000',ours,pachi,args.external_pairs,50100)
        katago=('gtp:data/katago/src/cpp/build/katago gtp -model data/katago/b18c384nbt-humanv0.bin.gz '
                '-human-model data/katago/b18c384nbt-humanv0.bin.gz -config data/katago/human.cfg '
                '-override-config humanSLProfile=rank_1d,maxVisits=8,nnCacheSizePowerOfTwo=14,searchRandSeed={seed}')
        match('katago_rank1d',ours,katago,args.external_pairs,51100)
        state.update(status='done',finished=datetime.now().isoformat())
        atomic(state_path,state)
        print('ALL DONE',ours,flush=True)
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}')
        atomic(state_path,state)
        raise

if __name__=='__main__':main()
