"""Evaluate value nets on the same held-out games: old test sources and the new generated set.

Test games come from ValueData(seed 0, 3%): the 'gen' source is first in both the old value_own
split and the new --gen-only split, so its test games are identical; 'rl'/'human' are value_own's
old test games; 'gen2' is the new 1.5M-game set (also reported for the s_{U+1} position only).
"""
import json, sys
import numpy as np, torch
from alphago.gtp import load_net
from alphago.models import pick_device
from alphago.value_data import ValueData, load_gen, load_rl, load_human

nets = sys.argv[1:] or ['checkpoints/value_own/best.pt', 'checkpoints/value_gen_only/best.pt']
dev = pick_device()
old = ValueData([load_gen(), load_rl(), load_human()])
new = ValueData([load_gen(), load_gen('data/value/gen_rlcont/batch_*.npz', name='gen2')])
g2 = new.sources[1]
sets = {}
for name, data, si in [('gen (기존 생성)', old, 0), ('rl (RL 학습 기록)', old, 1), ('human (KGS)', old, 2), ('gen2 (새 생성)', new, 1)]:
    rows = data.index['test'][data.index['test'][:, 0] == si]
    rows = rows[np.random.default_rng(3).permutation(len(rows))[:20000]]
    sets[name] = (data, rows)
rows = new.index['test'][new.index['test'][:, 0] == 1]
first = rows[rows[:, 2] == np.array([int(g2.plies[g][0]) for g in rows[:, 1]])]
sets['gen2 U번째 수 직후만'] = (new, first[np.random.default_rng(4).permutation(len(first))[:20000]])

@torch.no_grad()
def mse(net, data, rows):
    v, z = [], []
    for i in range(0, len(rows), 512):
        x, zz, _ = data.batch_from_index(rows[i:i + 512])
        v.append(net.forward_all(torch.from_numpy(x).to(dev).float())[1].cpu().numpy()); z.append(zz)
    v, z = np.concatenate(v), np.concatenate(z); k = z != 0
    return float(((v[k] - z[k]) ** 2).mean()), float((np.sign(v[k]) == z[k]).mean()), int(k.sum())

out = {}
for p in nets:
    net = load_net(p, dev); out[p] = {}
    for name, (data, rows) in sets.items():
        m, acc, n = mse(net, data, rows); out[p][name] = {'mse': round(m, 4), 'winner_acc': round(acc, 4), 'n': n}
print(f'{"테스트셋":<22}' + ''.join(f'{p.split("/")[1]:>26}' for p in nets))
for name in sets:
    print(f'{name:<22}' + ''.join(f'{out[p][name]["mse"]:>14.4f} ({out[p][name]["winner_acc"]:.1%})' for p in nets))
json.dump(out, open('runs/compare_value_nets.json', 'w'), indent=1, ensure_ascii=False)
