"""Atomic local archives of completed browser games and move provenance."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from alphago import Board
from alphago.gnugo import ownership


def search_record(root):
    if root is None or root.N.sum() <= 0:
        return None
    counts = np.zeros(362, np.int32)
    counts[root.moves] = np.rint(root.N).astype(np.int32)
    n = int(counts.sum())
    return {'visits': n, 'counts': counts.tolist(),
            'q': float(root.W.sum() / root.N.sum())}


def atomic_write(path, writer):
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def save_finished_game(directory, game_id, moves, records, komi, verdict, model):
    """JSON is written last as completion marker. Never invent absent MCTS targets."""
    if len(moves) != len(records):
        raise ValueError('Move provenance must align with the entire history')
    b = Board(19, komi, superko=True)
    features = []
    for a in moves:
        features.append(b.features().copy())
        b.play(a)
    if not b.game_over:
        raise ValueError('Only completed games may be auto-saved')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(np.asarray(moves, dtype='<i2').tobytes()).hexdigest()
    stem = directory / f'game_{game_id}_{digest[:12]}'
    paths = [stem.with_suffix(s) for s in ('.npz', '.sgf', '.json')]
    if all(p.exists() for p in paths):
        return str(paths[1])
    n = len(moves)
    policy = np.zeros((n, 362), np.float32)
    visits = np.zeros(n, np.int32)
    values = np.full(n, np.nan, np.float32)
    mask = np.zeros(n, bool)
    for i, rec in enumerate(records):
        search = rec.get('search')
        if search and search['visits'] > 0:
            counts = np.asarray(search['counts'], np.float64)
            if counts.shape != (362,) or (counts < 0).any() or counts.sum() != search['visits']:
                raise ValueError('Invalid recorded visit distribution')
            policy[i] = counts / counts.sum()
            visits[i], values[i] = search['visits'], search['q']
            # Human choices (even with analysis), imported moves and policy-only moves
            # are not relabeled as MCTS-selected policy training examples.
            mask[i] = rec['source'] == 'engine_mcts'
    judged = verdict is not None and np.isfinite(verdict['score'])
    score = float(verdict['score']) if judged else float('nan')
    dead = verdict['dead'] if judged else []
    own = ownership(moves, dead, 19, komi) if judged else np.zeros(361, np.int8)
    arrays = dict(moves=np.asarray(moves, np.int16), features=np.asarray(features, np.uint8),
                  policy=policy, search_values=values, visits=visits, train_mask=mask,
                  board_size=19, komi=komi, capped=False, judged=judged, score=score,
                  ownership=own, dead=np.asarray(dead, np.int16), seed=int(game_id[:12], 16),
                  source='browser_game', game_id=game_id)
    atomic_write(paths[0], lambda stream: np.savez_compressed(stream, **arrays))

    def escape(s):
        return str(s).replace('\\', '\\\\').replace(']', '\\]')
    result = 'Void' if not judged else ('0' if score == 0 else f'{"B" if score > 0 else "W"}+{abs(score):g}')
    sgf = f'(;GM[1]FF[4]CA[UTF-8]SZ[19]KM[{komi:g}]RU[Chinese]RE[{result}]C[{escape(model)}]'
    for i, (a, rec) in enumerate(zip(moves, records)):
        point = '' if a == 361 else chr(97 + a % 19) + chr(97 + 18 - a // 19)
        comment = rec['source']
        if rec.get('search'):
            comment += f"; visits={rec['search']['visits']}; Q={rec['search']['q']:.6f} (side to move)"
        sgf += f';{"B" if i % 2 == 0 else "W"}[{point}]C[{escape(comment)}]'
    sgf += ')'
    atomic_write(paths[1], lambda stream: stream.write(sgf.encode('utf-8')))
    metadata = {'game_id': game_id, 'source': 'browser_game', 'model': model,
                'moves': len(moves), 'komi': komi, 'judged': bool(judged),
                'score': score if judged else None, 'result': result, 'moves_sha256': digest,
                'training_positions': int(mask.sum()) if judged else 0, 'move_records': records,
                'note': 'Not pure self-play. Human/imported moves are not policy targets. No automatic training ingestion.'}
    atomic_write(paths[2], lambda stream: stream.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')))
    return str(paths[1])
