"""T5 fixed test: does each engine still waste a move on T5 (white group already alive per KataGo)?"""
import json, sys
import numpy as np, torch
from alphago import Board
from alphago.gtp import load_net, vertex_to_action, action_to_vertex
from alphago.mcts import NetEvaluator, MCTS
s = json.load(open('runs/diagnostics/t5_20261019/position.json'))
dev = torch.device('mps'); out = {}
for spec in sys.argv[1:]:
    pol, _, val = spec.partition('+')
    ev = NetEvaluator(load_net(pol, dev), load_net(val, dev) if val else None, device=dev)
    b = Board(19, 7.5, superko=True)
    for v in s['history']: b.play(vertex_to_action(v, 19))
    _, value = ev([b]); root = MCTS(ev, playouts=1600, batch=32).search(b)
    order = np.argsort(-root.N)[:4]
    top = [(action_to_vertex(int(root.moves[i]), 19), int(root.N[i]), round(float(root.W[i] / root.N[i]), 3)) for i in order]
    t5 = next((t for t in top if t[0] == 'T5'), None)
    out[spec] = {'white_value_net': round(float(value[0]), 3), 'top4_visits_q': top}
    print(f'{spec}\n  가치망 백 기대값 {value[0]:+.3f} · 1600회 상위: {top}  · T5 {"1위" if top[0][0]=="T5" else "1위 아님"}')
json.dump(out, open('runs/diagnostics/t5_20261019/engines_20261022.json', 'w'), indent=1)
