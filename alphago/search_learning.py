"""Search-policy iteration primitives, retaining separate policy and value nets.

Policy target: root visit distribution, not the sampled action or a win label.
Value target: adjudicated terminal outcome in the player-to-move perspective.
Capped/failed games are saved for diagnosis but never used as training examples.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alphago import Board
from alphago.data import symmetry_tables, transform
from alphago.mcts import MCTS


@dataclass
class SearchPlayConfig:
    visits: int = 400
    batch: int = 32
    board_size: int = 19
    komi: float = 7.5
    max_moves: int = 600
    noise_fraction: float = .25
    noise_alpha: float = .03
    temperature_moves: int = 30
    late_temperature: float = .2
    fast_visits: int = 100
    full_search_probability: float = 1.


def visit_target(root, actions):
    counts = np.asarray(root.N, np.float64)
    if not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0:
        raise ValueError('Invalid search visit counts')
    target = np.zeros(actions, np.float32)
    target[root.moves] = counts / counts.sum()
    return target


def choose_action(target, rng, temperature):
    if temperature <= 0:
        return int(np.argmax(target))
    support = np.flatnonzero(target > 0)
    logits = np.log(target[support].astype(np.float64)) / temperature
    weights = np.exp(logits - logits.max())
    return int(rng.choice(support, p=weights / weights.sum()))


def search_turn(board, evaluator, config, rng, noisy=True):
    if config.visits <= 0 or config.batch <= 0:
        raise ValueError('Search needs positive visits and batch')
    search = MCTS(evaluator, playouts=config.visits, batch=config.batch)
    root = search._expand([board])[0]
    if noisy and config.noise_fraction:
        noise = rng.dirichlet(np.full(len(root.P), config.noise_alpha))
        root.P = ((1 - config.noise_fraction) * root.P + config.noise_fraction * noise).astype(np.float32)
    root = search.search(board, root=root)
    count = int(root.N.sum())
    if count != config.visits:
        raise RuntimeError(f'Incomplete search: {count} != {config.visits}')
    target = visit_target(root, board.size**2 + 1)
    value = float(root.W.sum() / count)
    temperature = 1. if board.move_number < config.temperature_moves else config.late_temperature
    action = choose_action(target, rng, temperature if noisy else 0.)
    return action, target, value, count


def generate_game(evaluator, config, rng, judge, on_move=None):
    if not 0 < config.full_search_probability <= 1 or config.fast_visits <= 0:
        raise ValueError('Invalid randomized search budget')
    board = Board(config.board_size, config.komi, superko=True)
    moves, policies, values, visits, features, train_mask = [], [], [], [], [], []
    while not board.game_over and len(moves) < config.max_moves:
        features.append(board.features().copy())
        full = config.full_search_probability == 1 or rng.random() < config.full_search_probability
        turn_config = config if full else replace(config, visits=config.fast_visits)
        action, pi, value, count = search_turn(board, evaluator, turn_config, rng, noisy=full)
        train_mask.append(full)
        policies.append(pi); values.append(value); visits.append(count)
        board.play(action); moves.append(action)
        if on_move is not None:
            on_move(len(moves), count)
    record = {'moves': np.asarray(moves, np.int16), 'policy': np.asarray(policies, np.float32),
              'features': np.asarray(features, np.uint8),
              'train_mask': np.asarray(train_mask, np.bool_),
              'search_values': np.asarray(values, np.float32), 'visits': np.asarray(visits, np.int32),
              'board_size': config.board_size, 'komi': config.komi,
              'capped': not board.game_over, 'judged': False, 'score': float('nan'),
              'ownership': np.zeros(config.board_size**2, np.int8), 'dead': np.empty(0, np.int16)}
    if not record['capped']:
        try:
            score, dead, own = judge(moves)
            own = np.asarray(own, np.int8).reshape(-1)
            if not np.isfinite(score) or own.shape != (config.board_size**2,):
                raise ValueError('Invalid referee result')
            record.update(judged=True, score=float(score), dead=np.asarray(dead, np.int16), ownership=own)
        except Exception as exc:
            record['judge_error'] = repr(exc)
    return record


def save_game(path, record):
    path = Path(path)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **record)
    tmp.replace(path)


class SearchData:
    def __init__(self, paths, board_size=19):
        self.board_size = board_size
        self.games, rejected = [], 0
        for path in paths:
            with np.load(path, allow_pickle=False) as d:
                if bool(d['capped']) or not bool(d['judged']) or not np.isfinite(d['score']):
                    rejected += 1
                    continue
                if int(d['board_size']) != board_size:
                    raise ValueError('Board-size mismatch')
                moves, pi, visits = d['moves'].copy(), d['policy'].copy(), d['visits'].copy()
                if pi.shape != (len(moves), board_size**2 + 1) or len(visits) != len(moves):
                    raise ValueError('Position/target length mismatch')
                if not np.isfinite(pi).all() or (pi < 0).any() or not np.allclose(pi.sum(1), 1, atol=1e-5):
                    raise ValueError('Invalid policy targets')
                if len(moves) < 2 or not np.all(moves[-2:] == board_size**2):
                    raise ValueError('Adjudicated training game must terminate with two passes')
                features = d['features'].copy()
                if features.shape != (len(moves), 34, board_size, board_size):
                    raise ValueError('Stored features must match actual superko-aware search positions')
                mask = d['train_mask'].copy() if 'train_mask' in d else np.ones(len(moves), dtype=bool)
                if mask.dtype != np.bool_ or mask.shape != (len(moves),):
                    raise ValueError('Invalid full-search training mask')
                self.games.append({'moves': moves, 'policy': pi, 'features': features, 'train_mask': mask, 'score': float(d['score']),
                                   'ownership': d['ownership'].copy(), 'path': str(path)})
        self.rejected = rejected
        if not self.games:
            raise ValueError('No complete adjudicated games available for training')
        self.moves = np.concatenate([g['moves'] for g in self.games])
        self.offsets = np.r_[0, np.cumsum([len(g['moves']) for g in self.games])].astype(np.int64)
        self.rows = np.asarray([(i, p) for i, g in enumerate(self.games) for p in np.flatnonzero(g['train_mask'])], np.int64).reshape(-1, 2)
        if not len(self.rows):
            raise ValueError('No full-search training positions available')
        self.perm = symmetry_tables(board_size)

    def batch(self, rows, rng=None, augment=False):
        game, ply = rows[:, 0], rows[:, 1]
        features = np.stack([self.games[g]['features'][p] for g, p in rows])
        pi = np.stack([self.games[g]['policy'][p] for g, p in rows])
        side = np.where(ply % 2 == 0, 1., -1.).astype(np.float32)
        z = np.asarray([np.sign(self.games[g]['score']) for g in game], np.float32) * side
        own = np.stack([self.games[g]['ownership'] for g in game]).reshape(-1, self.board_size, self.board_size).astype(np.float32)
        own *= side[:, None, None]
        if augment:
            syms = rng.integers(0, 8, len(rows))
            for s in range(1, 8):
                selected = np.flatnonzero(syms == s)
                if len(selected):
                    features[selected] = transform(features[selected], s)
                    old = pi[selected].copy()
                    pi[np.ix_(selected, self.perm[s])] = old
                    own[selected] = transform(own[selected], s)
        return features, pi, z, own


def losses(policy, value, anchor, batch, device, kl_weight=.05, ownership_weight=1.5):
    features, pi, z, own = batch
    x = torch.from_numpy(features).to(device).float()
    target = torch.from_numpy(pi).to(device)
    logits, _ = policy(x)
    _, v, ownership = value.forward_all(x)
    logp = F.log_softmax(logits, dim=-1)
    policy_loss = -(target * logp).sum(-1).mean()
    value_loss = F.mse_loss(v, torch.from_numpy(z).to(device))
    with torch.no_grad():
        anchor_logits, _ = anchor(x)
        reference = F.softmax(anchor_logits, dim=-1)
    kl = F.kl_div(logp, reference, reduction='batchmean')
    own_loss = F.mse_loss(ownership, torch.from_numpy(own).to(device)) if ownership is not None else v.sum() * 0
    total = policy_loss + value_loss + kl_weight * kl + ownership_weight * own_loss
    return total, {'policy_ce': policy_loss, 'value_mse': value_loss, 'anchor_kl': kl, 'ownership_mse': own_loss}
