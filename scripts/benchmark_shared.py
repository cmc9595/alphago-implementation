"""Same positions/batch/search visits: one shared net versus two original nets."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','.10')
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from alphago import Board
from alphago.gtp import load_net
from alphago.mcts import MCTS,NetEvaluator


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.mps.set_per_process_memory_fraction(.12)
    p=load_net('checkpoints/sl_6x96/final.pt');v=load_net('checkpoints/value_own/best.pt');shared=load_net(args.model)
    evaluators={'separate':NetEvaluator(p,v),'shared':NetEvaluator(shared)}
    # Freeze actual game positions, not just the empty board.
    with np.load('data/value/gen/batch_00000.npz') as d:
        moves=d['moves'][d['offsets'][0]:d['offsets'][1]]
    boards=[]
    for ply in (0,30,60,90,120,150,180,210):
        b=Board(19,7.5,superko=True)
        for a in moves[:ply]:b.play(int(a))
        if not b.game_over:boards.append(b)
    batch=[boards[i%len(boards)] for i in range(32)]
    for evaluator in evaluators.values():
        evaluator(batch);evaluator(batch)
    records={name:{'batch32_seconds':[],'mcts400_seconds':[]} for name in evaluators}
    for repeat in range(4):
        for name in (('separate','shared') if repeat%2==0 else ('shared','separate')):
            evaluator=evaluators[name]
            t=time.perf_counter();evaluator(batch)
            records[name]['batch32_seconds'].append(time.perf_counter()-t)
    for i,b in enumerate(boards):
        for name in (('separate','shared') if i%2==0 else ('shared','separate')):
            t=time.perf_counter();root=MCTS(evaluators[name],playouts=400,batch=32).search(b)
            assert int(root.N.sum())==400
            records[name]['mcts400_seconds'].append(time.perf_counter()-t)
    medians={name:{key:float(np.median(values)) for key,values in r.items()} for name,r in records.items()}
    out={'batch':32,'search_visits':400,'positions':len(boards),'medians':medians,'samples':records,
         'inference_speedup':medians['separate']['batch32_seconds']/medians['shared']['batch32_seconds'],
         'mcts_speedup':medians['separate']['mcts400_seconds']/medians['shared']['mcts400_seconds'],
         'note':'Inference uses identical boards. MCTS traverses different trees; speed does not establish strength. Concurrent UI usage can affect timing.'}
    path=Path(args.out);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2));tmp.replace(path)
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
