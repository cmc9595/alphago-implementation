"""Re-run GNU Go adjudication for games saved with judged=False (e.g. referee failures). Writes in place."""
import glob, json, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
sys.path.insert(0, 'scripts')
from search_policy_iteration import Referee
from alphago.search_learning import save_game

run = sys.argv[1]; local = threading.local(); refs = []
def work(path):
    with np.load(path) as d:
        rec = {k: d[k] for k in d.files}
    if bool(rec['judged']) or bool(rec['capped']):
        return 0
    for attempt in range(2):  # GNU Go occasionally segfaults on a position; restart once, then skip
        if not hasattr(local, 'r') or local.r is None:
            local.r = Referee(float(rec['komi'])); refs.append(local.r)
        try:
            score, dead, own = local.r.judge(rec['moves'].tolist()); break
        except Exception:
            try: local.r.close()
            except Exception: pass
            local.r = None
    else:
        return 0
    rec.update(judged=np.bool_(True), score=np.float64(score), dead=np.asarray(dead, np.int16),
               ownership=np.asarray(own, np.int8).reshape(-1))
    save_game(path, rec)
    j = path.replace('.npz', '.json'); info = json.load(open(j))
    info.update(valid=True, training_positions=int(rec['train_mask'].sum()), rejudged=True)
    json.dump(info, open(j, 'w'))
    return 1
paths = sorted(glob.glob(f'runs/experiments/{run}/game_*.npz'))
with ThreadPoolExecutor(8) as ex:
    n = sum(ex.map(work, paths))
for r in refs:
    try: r.close()
    except Exception: pass
print('rejudged', n)
