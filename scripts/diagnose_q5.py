"""Q5 life/death diagnosis. All saved probabilities are WHITE's perspective."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','0.3')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','0.25')
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from alphago import Board,Color
from alphago.gtp import load_net,vertex_to_action,action_to_vertex
from alphago.mcts import MCTS,NetEvaluator,TreeBudget
from diagnose_ladder import atomic,white_value,CountEvaluator,snapshot

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'runs/diagnostics/q5_20261018'
CANDIDATES=['R5','S5','T6','T7','T5','Q13','B9']


def main():
    source=json.loads((OUT/'position.json').read_text())
    assert len(source['history'])==158 and source['history'][-1]=='S9'
    b=Board(19,source['komi'],superko=True)
    for v in source['history']:b.play(vertex_to_action(v,19))
    assert b.to_move==Color.BLACK
    modeldir=OUT/'models';modeldir.mkdir(exist_ok=True)
    paths={}
    for key in ['sl','value']:
        dest=modeldir/(key+'.pt')
        if not dest.exists():shutil.copy2(ROOT/'runs/experiments/fixed_visits_20261017/models'/(key+'.pt'),dest)
        paths[key]=str(dest)
    manifest={'position_sha256':hashlib.sha256((OUT/'position.json').read_bytes()).hexdigest(),
              'models':{k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in paths.items()},
              'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['alphago/mcts.py','scripts/diagnose_q5.py','scripts/diagnose_ladder.py']},
              'batch':32,'candidates':CANDIDATES,'normal_targets':[400,1600,6400,25600], 'forced_targets':[400,1600,6400]}
    atomic(OUT/'plan.json',manifest)
    policy=load_net(paths['sl']);value=load_net(paths['value'])
    evaluators={'sl_value':NetEvaluator(policy),'new_value':NetEvaluator(policy,value)}
    result={'position_ply':158,'side_to_move':'B','perspective':'WHITE','batch':32,'raw':{},'normal':{},'forced':{}}
    for model,evaluator in evaluators.items():
        boards=[b];names=['before']
        for v in CANDIDATES:
            a=vertex_to_action(v,19)
            if b.is_legal(a):
                c=b.copy();c.play(a);boards.append(c);names.append(v)
        # A direct test of overlooking T7: Black Q13, White T7.
        c=b.copy();c.play(vertex_to_action('Q13',19));c.play(vertex_to_action('T7',19));boards.append(c);names.append('Q13_T7')
        logits,values=evaluator(boards)
        pp=np.exp(logits[0]-logits[0].max());pp/=pp.sum()
        result['policy_exact']={v:float(pp[vertex_to_action(v,19)]) for v in CANDIDATES}
        result['raw'][model]={name:white_value(board,v) for name,board,v in zip(names,boards,values)}
        branches=[None] if model=='sl_value' else [None]+CANDIDATES
        for forced in branches:
            atomic(OUT/'state.json',{'status':'running','model':model,'branch':forced or 'normal'})
            count=CountEvaluator(evaluator);root=None;budget=TreeBudget();elapsed=0.;previous=0;rows=[]
            if forced:
                root=MCTS(count)._expand([b])[0];a=vertex_to_action(forced,19)
                root.moves=np.array([a],np.int64);root.P=np.ones(1,np.float32);root.N=np.zeros(1,np.float32);root.W=np.zeros(1,np.float32);budget.add_root(root)
            for target in manifest['forced_targets' if forced else 'normal_targets']:
                t=time.perf_counter();root=MCTS(count,playouts=target-previous,batch=32).search(b,root=root,budget=budget)
                elapsed+=time.perf_counter()-t;previous=target
                row=snapshot(b,root,budget,elapsed,count);row['tracked']={}
                for v in CANDIDATES:
                    ids=np.flatnonzero(root.moves==vertex_to_action(v,19))
                    if len(ids):
                        i=int(ids[0]);n=int(root.N[i]);row['tracked'][v]={'visits':n,'prior':float(root.P[i]),'white_win':white_value(b,root.W[i]/n) if n else None}
                rows.append(row)
                if forced:result['forced'].setdefault(model,{})[forced]=rows
                else:result['normal'][model]=rows
                atomic(OUT/'own.json',result)
                print(json.dumps({'model':model,'branch':forced or 'normal','visits':target,'best':row['candidates'][0]['move'],'black_win':round(1-row['root_white_win'],4)}),flush=True)
            del root
    atomic(OUT/'state.json',{'status':'own_done'})
    print('OWN Q5 DONE',flush=True)


if __name__=='__main__':main()
