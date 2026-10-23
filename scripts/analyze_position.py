"""Reproduce life/death evaluation around selected plies of an SGF or web history JSON.

python scripts/analyze_position.py game.sgf --plies 120,121,122
Output compares raw value and MCTS 400/1600/6400, from Black's perspective.
A value estimate is not an adjudicated life/death verdict.
"""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','0.3')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','0.25')
import argparse
import json
from pathlib import Path
import time
import torch
from alphago import Board, Color
from alphago.gtp import load_net, vertex_to_action, action_to_vertex
from alphago.mcts import MCTS, NetEvaluator
from alphago.sgf import parse_games


def read_game(path):
    if path.suffix.lower()=='.sgf':
        games=parse_games(path.read_text())
        if len(games)!=1: raise ValueError('Select a single SGF game.')
        game=games[0]
        if game.size!=19: raise ValueError('This model requires a 19x19 board.')
        return game.komi if game.komi is not None else 7.5, game.setup_black, game.setup_white, game.moves
    data=json.loads(path.read_text())
    return float(data.get('komi',7.5)),[],[],[(1 if i%2==0 else 2,vertex_to_action(v,19)) for i,v in enumerate(data['history'])]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('record',type=Path)
    ap.add_argument('--plies',help='comma-separated move counts; default: last three positions')
    ap.add_argument('--visits',default='400,1600,6400')
    ap.add_argument('--policy',default='checkpoints/sl_6x96/final.pt')
    ap.add_argument('--value',default='checkpoints/value_own/best.pt')
    ap.add_argument('--out',type=Path,default=Path('runs/position_checks/latest.json'))
    args=ap.parse_args()
    komi,black,white,moves=read_game(args.record)
    plies=sorted(set(map(int,args.plies.split(',')))) if args.plies else list(range(max(0,len(moves)-2),len(moves)+1))
    visits=list(map(int,args.visits.split(',')))
    if not plies or min(plies)<0 or max(plies)>len(moves) or not visits or min(visits)<1:
        ap.error('plies must be within the record; visits must be positive')
    policy=load_net(args.policy); value=load_net(args.value)
    evaluators={'sl_value':NetEvaluator(policy), 'new_value':NetEvaluator(policy,value)}
    board=Board(19,komi,superko=True)
    for a in black: board.place_stone(a,Color.BLACK)
    for a in white: board.place_stone(a,Color.WHITE)
    result={'record':str(args.record),'policy':args.policy,'value':args.value,'komi':komi,
            'note':'Black win estimates, not GNU Go adjudication. SL policy fixed in both comparisons.', 'positions':[]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for ply in range(max(plies)+1):
        if ply:
            colour,action=moves[ply-1]
            if board.game_over: raise ValueError('Record continues after two passes')
            board.set_to_move(Color.BLACK if colour==1 else Color.WHITE)
            board.play(action)
        if ply not in plies: continue
        row={'ply':ply,'game_over':board.game_over,'models':{}}
        for name,evaluator in evaluators.items():
            _,raw=evaluator([board]); sign=1 if board.to_move==Color.BLACK else -1
            entry={'raw_black_win':float((1+sign*raw[0])/2),'search':[]}
            if not board.game_over:
                for n in visits:
                    started=time.perf_counter(); root=MCTS(evaluator,playouts=n).search(board)
                    mean=float(root.W.sum()/root.N.sum())
                    candidates=[]
                    for c in MCTS.summary(root,6):
                        candidates.append({**c,'vertex':action_to_vertex(c['action'],19),'black_win':(1+sign*c['q'])/2})
                    entry['search'].append({'visits':int(root.N.sum()),'seconds':time.perf_counter()-started,
                        'root_black_win':(1+sign*mean)/2,'candidates':candidates})
            row['models'][name]=entry
        result['positions'].append(row)
        tmp=args.out.with_suffix('.tmp'); tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False)); tmp.replace(args.out)
        print(json.dumps({'ply':ply,'raw_black_win':{k:v['raw_black_win'] for k,v in row['models'].items()} }),flush=True)
    print('Saved',args.out,flush=True)


if __name__=='__main__': main()
