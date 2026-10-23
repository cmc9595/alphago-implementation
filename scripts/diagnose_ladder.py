"""Targeted ladder diagnosis: fixed SL policy, compare raw and searched values.

No live-board modifications. All probabilities are White's perspective.
python scripts/diagnose_ladder.py
"""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','0.3')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','0.25')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from alphago import Board, Color
from alphago.gtp import load_net, vertex_to_action, action_to_vertex
from alphago.mcts import MCTS, NetEvaluator, TreeBudget

ROOT=Path(__file__).resolve().parent.parent


def atomic(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)); tmp.replace(path)


def white_value(board,value):
    return (1+float(value)*(1 if board.to_move==Color.WHITE else -1))/2


def principal_variation(root,first=None,limit=12):
    pv=[]; node=root
    while node is not None and node.N.sum()>0 and len(pv)<limit:
        i=first if not pv and first is not None else int(np.argmax(node.N))
        pv.append(action_to_vertex(int(node.moves[i]),19)); node=node.children.get(i)
    return pv


class CountEvaluator:
    def __init__(self,evaluator): self.evaluator=evaluator; self.positions=0; self.calls=0
    def __call__(self,boards):
        self.positions+=len(boards); self.calls+=1
        return self.evaluator(boards)


def snapshot(board,root,budget,seconds,count):
    total=float(root.N.sum())
    rows=[]
    for i in np.argsort(-root.N)[:8]:
        n=int(root.N[i]); q=float(root.W[i]/n) if n else None
        rows.append({'move':action_to_vertex(int(root.moves[i]),19),'visits':n,
                     'visit_share':n/total if total else 0,'white_win':white_value(board,q) if q is not None else None,
                     'prior':float(root.P[i]),'pv':principal_variation(root,int(i)) if n else []})
    tracked={}
    for v in ['H9','M3','M4']:
        ids=np.flatnonzero(root.moves==vertex_to_action(v,19))
        if len(ids):
            i=int(ids[0]); n=int(root.N[i]); tracked[v]={'visits':n,'white_win':white_value(board,root.W[i]/n) if n else None}
    return {'visits':int(total),'nodes':budget.nodes,'estimated_tree_bytes':budget.bytes,
            'seconds':seconds,'neural_positions':count.positions,'neural_batches':count.calls,
            'root_white_win':white_value(board,root.W.sum()/total) if total else white_value(board,root.value),
            'candidates':rows,'tracked':tracked}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',type=Path,default=ROOT/'runs/position_checks/ladder_20261018.json')
    ap.add_argument('--out',type=Path,default=ROOT/'runs/diagnostics/ladder_20261018')
    ap.add_argument('--batch',type=int,default=32)
    args=ap.parse_args();os.chdir(ROOT);out=args.out;out.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.input.read_text())
    assert len(source['history'])==73 and source['history'][-1]=='L4', 'This named diagnosis is for the saved 73-ply position.'
    atomic(out/'position.json',source)
    paths={'sl':ROOT/'runs/experiments/fixed_visits_20261017/models/sl.pt',
           'value':ROOT/'runs/experiments/fixed_visits_20261017/models/value.pt'}
    modeldir=out/'models';modeldir.mkdir(exist_ok=True)
    for key,path in list(paths.items()):
        dest=modeldir/(key+'.pt')
        if not dest.exists():shutil.copy2(path,dest)
        paths[key]=dest
    manifest={'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),'batch':args.batch,
              'models':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
              'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['alphago/mcts.py','scripts/diagnose_ladder.py']},
              'root_targets':[400,1600,6400,25600],'forced_targets':[400,1600,6400]}
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=manifest:
        raise ValueError('Diagnosis conditions changed; use a different --out directory.')
    atomic(out/'plan.json',manifest);shutil.copy2(ROOT/'alphago/mcts.py',out/'mcts_source.py')
    b=Board(19,source['komi'],superko=True)
    for v in source['history']:b.play(vertex_to_action(v,19))
    assert b.to_move==Color.WHITE
    policy=load_net(str(paths['sl']));value=load_net(str(paths['value']))
    evaluators={'sl_value':NetEvaluator(policy),'new_value':NetEvaluator(policy,value)}
    result={'position_ply':73,'side_to_move':'W','perspective':'WHITE','batch':args.batch,
            'komi':source['komi'],'raw':{},'normal':{},'forced':{}}
    for model,evaluator in evaluators.items():
        boards=[b];names=['before']
        for move in ['H9','M3','M4']:
            c=b.copy();c.play(vertex_to_action(move,19));boards.append(c);names.append(move)
        _,values=evaluator(boards)
        result['raw'][model]={name:white_value(c,v) for name,c,v in zip(names,boards,values)}
        atomic(out/'own.json',result)
        for forced in [None,'H9','M3','M4']:
            key=forced or 'normal';state={'status':'running','model':model,'branch':key}
            atomic(out/'state.json',state)
            count=CountEvaluator(evaluator);root=None;budget=TreeBudget();elapsed=0.;rows=[]
            if forced:
                root=MCTS(count)._expand([b])[0]
                a=vertex_to_action(forced,19)
                root.moves=np.array([a],np.int64);root.P=np.ones(1,np.float32)
                root.N=np.zeros(1,np.float32);root.W=np.zeros(1,np.float32)
                budget.add_root(root)
            previous=0
            for target in manifest['forced_targets' if forced else 'root_targets']:
                t=time.perf_counter();root=MCTS(count,playouts=target-previous,batch=args.batch).search(b,root=root,budget=budget)
                elapsed+=time.perf_counter()-t; previous=target
                rows.append(snapshot(b,root,budget,elapsed,count))
                if forced:result['forced'].setdefault(model,{})[forced]=rows
                else:result['normal'][model]=rows
                atomic(out/'own.json',result)
                print(json.dumps({'model':model,'branch':key,'visits':target,'best':rows[-1]['candidates'][0]['move'],
                                  'white_win':round(rows[-1]['root_white_win'],4)},ensure_ascii=False),flush=True)
            del root
    atomic(out/'state.json',{'status':'own_done'})
    print('OWN DONE',flush=True)


if __name__=='__main__':main()
