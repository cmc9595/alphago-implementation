"""Bounded game-uniform replay, stable splits, and an incremental compact cache."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from alphago.data import symmetry_tables, transform

CACHE_VERSION = 1


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def move_hash(moves):
    return hashlib.sha256(np.asarray(moves, dtype='<i2').tobytes()).hexdigest()


def validation_game(digest):
    # Permanent identity-based split; unaffected by source directory or window eviction.
    return int(hashlib.sha256(('replay-validation-v1:' + digest).encode()).hexdigest()[:16], 16) % 20 == 0


def cache_game(path, cache_dir):
    """Read only finalized files; retain only full-search positions, bit-packed."""
    path, cache_dir = Path(path), Path(cache_dir)
    stat = path.stat()
    identity = f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{CACHE_VERSION}'
    key = hashlib.sha256(identity.encode()).hexdigest()
    meta = cache_dir / key[:2] / f'{key}.json'
    if meta.exists():
        rec = json.loads(meta.read_text())
        if rec.get('reason') or Path(rec['cache']).exists():
            return rec
    rec = {'source': str(path.resolve()), 'key': key}
    with np.load(path) as d:
        rec['hash'] = move_hash(d['moves'])
        idx = np.flatnonzero(d['train_mask'])
        score = float(d['score'])
        if bool(d['capped']) or not bool(d['judged']) or not np.isfinite(score) or score == 0:
            rec['reason'] = 'invalid_or_unlabelled'
        elif not len(idx):
            rec['reason'] = 'no_full_search_positions'
        else:
            x = d['features'][idx]
            if x.shape[1:] != (34, 19, 19) or x.max() > 1:
                raise ValueError(f'Unexpected feature representation: {path}')
            pi = d['policy'][idx].astype(np.float16)
            if not np.isfinite(pi).all() or (pi < 0).any() or not np.allclose(pi.sum(1), 1, atol=.002):
                raise ValueError(f'Invalid policy target: {path}')
            side = np.where(idx % 2 == 0, 1, -1).astype(np.int8)
            target = cache_dir / key[:2] / f'{key}.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, suffix='.tmp')
            try:
                with os.fdopen(fd, 'wb') as stream:
                    # Uncompressed, compact arrays: no repeated full-game feature decompression.
                    np.savez(stream, x=np.packbits(x, axis=1), pi=pi,
                             z=(np.sign(score) * side).astype(np.float32),
                             own=d['ownership'].reshape(1, 19, 19).astype(np.int8) * side[:, None, None],
                             plies=idx.astype(np.int32))
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
            rec.update(cache=str(target.resolve()), positions=len(idx))
    atomic_json(meta, rec)
    return rec


def build_pool(sources, fixed_manifest, cache_dir, output, max_games=100000, latest=None, progress=None,
               validation_reference=None):
    """sources: oldest->newest {directory, policy}. Newest duplicate takes precedence."""
    excluded = {r['sha256_moves'] for r in json.loads(Path(fixed_manifest).read_text())['test_games']}
    if not excluded: raise ValueError('Empty permanent holdout')
    seen, train, validation, rejected = set(), [], [], {}
    processed = 0
    for source in reversed(sources):
        directory = Path(source['directory']).resolve()
        for path in sorted(directory.glob('game_*.npz'), key=lambda p: int(p.stem.split('_')[1]), reverse=True):
            if not path.with_suffix('.json').exists(): continue
            rec = cache_game(path, cache_dir)
            processed += 1
            if progress and processed % 1000 == 0:
                progress(processed)
            reason = rec.get('reason')
            if rec['hash'] in excluded: reason = 'fixed_holdout'
            if not reason and rec['hash'] in seen: reason = 'duplicate'
            if reason:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            seen.add(rec['hash'])
            row = {**rec, 'policy': bool(source['policy']),
                   'latest': directory == Path(latest).resolve() if latest else False}
            (validation if validation_game(rec['hash']) else train).append(row)
    # Source order is newest first. Keep recent whole games; never delete originals.
    selected = train[:max_games]
    validation.sort(key=lambda r: r['hash'])
    monitored = validation[:500] if validation_reference is None else validation_reference
    if any(r['hash'] in excluded or not validation_game(r['hash']) or not Path(r['cache']).exists() for r in monitored):
        raise ValueError('Invalid frozen internal validation reference')
    body = {'version': CACHE_VERSION, 'fixed_manifest_sha256': sha256(fixed_manifest),
            'max_games': max_games, 'train': selected, 'validation': monitored,
            'validation_games_total': len(validation), 'rejected': rejected,
            'available_train_games': len(train), 'latest_train_games': sum(r['latest'] for r in selected),
            'sources': sources}
    if not selected or not body['validation'] or not body['latest_train_games']:
        raise ValueError('Replay requires historical training, latest training and validation games')
    atomic_json(output, body)
    return body


class ReplaySampler:
    def __init__(self, pool, cache_games=256):
        self.pool = pool
        train_hashes = [r['hash'] for r in pool['train']]
        if len(set(train_hashes)) != len(train_hashes): raise ValueError('Duplicate training game')
        if set(train_hashes).intersection(r['hash'] for r in pool['validation']):
            raise ValueError('Validation game leaked into replay training')
        self.latest = [r for r in pool['train'] if r['latest']]
        self.history = [r for r in pool['train'] if not r['latest']]
        if not self.latest or not self.history: raise ValueError('Both replay strata are required')
        self.cache, self.cache_games = OrderedDict(), cache_games
        self.perm = symmetry_tables(19)

    def load(self, row):
        key = row['cache']
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        with np.load(key) as d:
            value = {k: d[k] for k in d.files}
        self.cache[key] = value
        if len(self.cache) > self.cache_games: self.cache.popitem(last=False)
        return value

    def sample(self, size, rng, latest_fraction=.5, value_fraction=.5, augment=True, uniform=False):
        rows = []
        if uniform:
            source = self.pool['train']
            if len(source) < size: raise ValueError('Not enough distinct games for this batch')
            rows = [source[i] for i in rng.choice(len(source), size, replace=False)]
        else:
            n = round(size * latest_fraction)
            for source, count in ((self.latest, n), (self.history, size - n)):
                if len(source) < count: raise ValueError('Not enough distinct games for this batch')
                rows.extend(source[i] for i in rng.choice(len(source), count, replace=False))
        rng.shuffle(rows)
        x, pi, z, own, pm, identities = [], [], [], [], [], []
        for row in rows:
            d = self.load(row)
            j = int(rng.integers(len(d['z'])))
            x.append(np.unpackbits(d['x'][j], axis=0, count=34))
            pi.append(d['pi'][j].astype(np.float32)); z.append(d['z'][j]); own.append(d['own'][j])
            pm.append(row['policy']); identities.append((row['hash'], int(d['plies'][j])))
        x, pi, z, own = np.stack(x), np.stack(pi), np.asarray(z, np.float32), np.asarray(own, np.float32)
        if augment:
            for i, sym in enumerate(rng.integers(0, 8, size)):
                if sym:
                    x[i] = transform(x[i], sym); own[i] = transform(own[i], sym)
                    old = pi[i].copy(); pi[i, self.perm[sym]] = old
        return x, pi, z, own, np.asarray(pm, bool), rng.random(size) < value_fraction, identities
