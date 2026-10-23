"""Paired validation of scalar-value symmetry averaging, plus equal-time MCTS cost."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.3')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '0.25')
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from alphago import Board
from alphago.data import transform
from alphago.gtp import load_net
from alphago.mcts import SYMMETRIES, MCTSPlayer
from alphago.value_data import ValueData, load_gen, load_rl, load_human


def measures(v,z):
    valid=z!=0
    v,z=v[valid],z[valid]
    p,t=(v+1)/2,(z+1)/2
    ece=0.
    for b in range(10):
        m=(p>=b/10)&(p<(b+1)/10 if b<9 else p<=1)
        if m.any(): ece+=m.mean()*abs(p[m].mean()-t[m].mean())
    return {'n':len(z),'mse':float(np.mean((v-z)**2)),
            'winner_acc':float(np.mean(np.sign(v)==z)), 'ece':float(ece)}


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--positions',type=int,default=8192)
    ap.add_argument('--seconds',type=float,default=.35)
    ap.add_argument('--out',default='runs/experiments/ensemble_20261017/diagnostics.json')
    ap.add_argument('--policy',default='checkpoints/sl_6x96/final.pt')
    ap.add_argument('--value',default='checkpoints/value_own/best.pt')
    args=ap.parse_args()
    data=ValueData([load_gen(),load_rl(),load_human()])
    rows=data.index['test'][np.random.default_rng(4817).permutation(data.size('test'))[:args.positions]]
    out={'positions':len(rows),'seconds_per_move':args.seconds,'models':{}}
    # D4 composition: output index after applying h to an input already transformed by g.
    idx=np.arange(9).reshape(3,3)
    signatures={transform(idx,k).tobytes():k for k in range(8)}
    comp=np.array([[signatures[transform(transform(idx,g),h).tobytes()] for h in range(8)] for g in range(8)])
    for name,path in [('sl',args.policy),('value_own',args.value)]:
        net=load_net(path)
        vals=[]; targets=[]; seconds=[]
        for i in range(0,len(rows),128):
            feats,z,_=data.batch_from_index(rows[i:i+128])
            targets.append(z)
            predictions=[]
            for s in range(8):
                x=torch.from_numpy(np.ascontiguousarray(transform(feats,s))).to(next(net.parameters()).device).float()
                t0=time.perf_counter()
                predictions.append(net(x)[1].cpu().numpy())
                seconds.append(time.perf_counter()-t0)
            vals.append(np.stack(predictions))
        raw=np.concatenate(vals,axis=1); z=np.concatenate(targets)
        modes={}
        for n,group in SYMMETRIES.items():
            v=raw[list(group)].mean(0)
            stat={'all':measures(v,z)}
            for si,src in enumerate(data.sources):
                m=rows[:,0]==si
                stat[src.name]=measures(v[m],z[m])
            for lo,hi,label in [(0,30,'opening'),(30,80,'early'),(80,150,'middle'),(150,1000,'late')]:
                m=(rows[:,2]>=lo)&(rows[:,2]<hi)
                stat[label]=measures(v[m],z[m])
            across=np.stack([raw[comp[g,list(group)]].mean(0) for g in range(8)])
            stat['orientation_win_probability_range_mean']=float(np.ptp((across+1)/2,axis=0).mean())
            stat['orientation_win_probability_range_p95']=float(np.quantile(np.ptp((across+1)/2,axis=0),.95))
            modes[str(n)]=stat
        out['models'][name]={'modes':modes,'raw_forward_seconds':sum(seconds)}
        print(json.dumps({'model':name,'modes':{k:{'mse':v['all']['mse'],'orientation_range':v['orientation_win_probability_range_mean']} for k,v in modes.items()}}),flush=True)
        del net
    policy=load_net(args.policy); value=load_net(args.value)
    # Fixed, legal opening positions; this is a speed diagnostic, not a strength test.
    boards=[]
    for moves in ([60,300,72,288],[72,300,60,288,40,316],[288,72,300,60,316,40,319,41]):
        b=Board(19,superko=True)
        for a in moves: b.play(a)
        boards.append(b)
    out['search_benchmark']={}
    for n in SYMMETRIES:
        p=MCTSPlayer(policy,value,playouts=20000,value_symmetries=n,time_limit=args.seconds)
        p.genmove(boards[0])
        times=[]; visits=[]
        for b in boards:
            t=time.perf_counter();p.genmove(b);times.append(time.perf_counter()-t);visits.append(float(p.last_root.N.sum()))
        out['search_benchmark'][str(n)]={'mean_seconds':float(np.mean(times)),'mean_visits':float(np.mean(visits))}
    modes=out['models']['value_own']['modes']
    candidate=min((2,4,8),key=lambda n:modes[str(n)]['gen']['mse'])
    out['candidate_symmetries']=candidate
    out['selection_note']='Candidate chosen on validation MSE only; equal-time paired matches decide adoption. Validation split reused; not an untouched final test.'
    path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(path)
    print('done',str(path),flush=True)

if __name__=='__main__':main()
