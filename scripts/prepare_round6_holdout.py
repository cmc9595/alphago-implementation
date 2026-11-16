"""After 15k games finish, create immutable-by-convention train/test copies.

Original self-play files are preserved. Train ONLY on the new train directory.
This script does not start training or GPU evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = 'selfplay_round6_1600v_p125_f200_seed20630000'
SEED = 20261114


def select_split(games, count, seed):
    """Split whole exact-duplicate groups together; test has unique games."""
    groups = {}
    for name, digest, plies in games:
        groups.setdefault(digest, []).append((name, plies))
    keys = sorted(groups)
    if len(keys) < count:
        raise ValueError('Not enough unique valid games')
    chosen = {keys[i] for i in np.random.default_rng(seed).permutation(len(keys))[:count]}
    test, train, duplicates = [], [], []
    for digest in keys:
        rows = groups[digest]
        if digest in chosen:
            name, plies = rows[0]
            test.append({'file': name, 'sha256_moves': digest, 'plies': plies})
            duplicates.extend(name for name, _ in rows[1:])
        else:
            train.extend(name for name, _ in rows)
    return test, train, duplicates


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--wait', action='store_true')
    args = ap.parse_args()
    source = ROOT / 'runs/experiments' / RUN
    out = ROOT / 'runs/datasets/round6_fixed1000'
    while len(list(source.glob('game_*.json'))) != 15000 or not (source / 'summary.json').exists():
        if not args.wait:
            raise RuntimeError('Wait until all 15,000 games and summary.json are saved')
        print('Waiting for all 15,000 games to finish', flush=True)
        time.sleep(60)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite fixed split: {out}')
    games = []
    rejected = []
    for path in sorted(source.glob('game_*.npz')):
        with np.load(path) as d:
            score = float(d['score'])
            plies = np.flatnonzero(d['train_mask']).tolist()
            if bool(d['capped']) or not bool(d['judged']) or not np.isfinite(score) or score == 0 or not plies:
                rejected.append(path.name)
                continue
            digest = hashlib.sha256(np.asarray(d['moves'], dtype='<i2').tobytes()).hexdigest()
        games.append((path.name, digest, plies))
    test, train, duplicates = select_split(games, 1000, SEED)
    for split, names in [('test', [r['file'] for r in test]), ('train', train)]:
        target = out / split
        target.mkdir(parents=True)
        for name in names:
            for suffix in ('.npz', '.json', '.sgf'):
                p = (source / name).with_suffix(suffix)
                if p.exists():
                    shutil.copy2(p, target / p.name)
    manifest = {'source': str(source.relative_to(ROOT)), 'seed': SEED,
                'test_games': test, 'train_games': train, 'rejected': rejected,
                'excluded_test_duplicates': duplicates,
                'evaluation': 'Fixed train_mask plies; side-to-move final outcome +/-1; no augmentation or MCTS',
                'warning': 'Train on this train/ directory ONLY, never on the original source containing test games.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'README.txt').write_text(
        'Fixed 1000-game value evaluation set. DO NOT TRAIN ON test/ OR THE ORIGINAL SOURCE.\n'
        'Use train/ for round-6 training; keep test/ out of ALL later training rounds.\n'
        'Use manifest.json test_games[].plies for identical model comparisons.\n'
        'Report both per-position MSE and mean per-game MSE against final outcome.\n')
    print(f'Prepared {len(test)} fixed test games and {len(train)} training games: {out}', flush=True)


if __name__ == '__main__':
    main()
