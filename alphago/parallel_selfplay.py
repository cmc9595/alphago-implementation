"""Many self-play games at once, sharing one large network batch per round.

Each game keeps its own PUCT tree (same rules as alphago.mcts.MCTS). Every round, each
active game contributes up to `leaves_per_game` leaves (virtual loss within its own tree),
all leaves from all games are evaluated in one call, then backed up. Compared with one game
and batch 32, the GPU sees large batches and each tree needs much less virtual loss.

Optional speed-ups (off by default, so data matches the sequential generator):
  reuse_tree   keep the chosen child's subtree as the next root for fast moves; its inherited
               visits count toward `fast_visits`. Full (training) moves always start fresh.
  resign       stop when the root value stays below -resign_threshold for resign_moves
               consecutive own moves; `no_resign_fraction` of games never resign so the
               false-resignation rate can be measured. Resigned games have no ownership.
"""
from dataclasses import dataclass, field
import time

import numpy as np

from alphago import Board, Color
from alphago.search_learning import SearchPlayConfig, visit_target, choose_action
from alphago.mcts import Node


@dataclass
class ParallelConfig:
    play: SearchPlayConfig = field(default_factory=SearchPlayConfig)
    parallel_games: int = 32
    leaves_per_game: int = 8
    max_eval_batch: int = 512
    c_puct: float = 1.5
    virtual_loss: float = 1.0
    fpu_reduction: float = 0.2
    reuse_tree: bool = False
    resign: bool = False
    resign_threshold: float = 0.80   # resign when root value < -threshold (≈10% win)
    resign_moves: int = 3
    no_resign_fraction: float = 0.1


def make_nodes(boards, logits, values):
    nodes = []
    for b, lg, v in zip(boards, logits, values):
        if b.game_over:
            nodes.append(Node(np.empty(0, np.int64), np.empty(0, np.float32), float(v)))
            continue
        moves = np.flatnonzero(b.sensible_mask())
        p = np.exp(lg[moves] - lg[moves].max())
        nodes.append(Node(moves, p / p.sum(), float(v)))
    return nodes


class Game:
    def __init__(self, gid, seed, cfg: ParallelConfig):
        self.gid, self.seed, self.cfg = gid, seed, cfg
        self.rng = np.random.default_rng(seed)
        p = cfg.play
        self.board = Board(p.board_size, p.komi, superko=True)
        self.moves, self.policies, self.values, self.visits, self.features, self.mask = [], [], [], [], [], []
        self.root = None
        self.low_streak = [0, 0]  # per colour: consecutive own moves below threshold
        self.allow_resign = cfg.resign and self.rng.random() >= cfg.no_resign_fraction
        self.resigned_winner = 0  # +1 black, -1 white
        self.started = time.monotonic()
        self.total_visits = 0
        self._new_turn()

    def _new_turn(self):
        p = self.cfg.play
        self.full = p.full_search_probability == 1 or self.rng.random() < p.full_search_probability
        self.target_visits = p.visits if self.full else p.fast_visits
        self.noised = False
        if self.full:
            self.root = None  # training targets must come from a fresh tree
        # Fast moves with a reused subtree: inherited visits count toward the budget.
        self.base_visits = 0

    @property
    def finished(self):
        return self.board.game_over or self.resigned_winner != 0 or len(self.moves) >= self.cfg.play.max_moves

    def done_visits(self):
        return 0 if self.root is None else int(self.root.N.sum()) - self.base_visits

    def select_leaves(self, count):
        """Descend with virtual loss. Returns [(path, board or None, cached terminal node or None)]."""
        cfg, out = self.cfg, []
        for _ in range(count):
            node, b, path = self.root, self.board.copy(), []
            while node.moves.size:
                total = node.N.sum()
                parent_q = (node.W.sum() / total) if total > 0 else node.value
                q = np.where(node.N > 0, node.W / np.maximum(node.N, 1e-9), parent_q - cfg.fpu_reduction)
                i = int(np.argmax(q + cfg.c_puct * node.P * np.sqrt(total + 1) / (1 + node.N)))
                path.append((node, i))
                node.N[i] += cfg.virtual_loss; node.W[i] -= cfg.virtual_loss
                b.play(int(node.moves[i]))
                child = node.children.get(i)
                if child is None:
                    out.append((path, b, None)); break
                node = child
            else:
                out.append((path, None, node))
        return out

    def backup(self, path, leaf):
        vl = self.cfg.virtual_loss
        parent, i = path[-1]
        parent.children.setdefault(i, leaf)
        val = -leaf.value
        for node, j in reversed(path):
            node.N[j] += 1 - vl; node.W[j] += val + vl
            val = -val

    def set_root(self, node):
        p = self.cfg.play
        if self.full and p.noise_fraction and not self.noised and node.moves.size:
            noise = self.rng.dirichlet(np.full(len(node.P), p.noise_alpha))
            node.P = ((1 - p.noise_fraction) * node.P + p.noise_fraction * noise).astype(np.float32)
            self.noised = True
        self.root = node

    def play_move(self):
        p, root = self.cfg.play, self.root
        target = visit_target(root, p.board_size ** 2 + 1)
        n = int(root.N.sum())
        value = float(root.W.sum() / n)
        temperature = (1. if self.board.move_number < p.temperature_moves else p.late_temperature) if self.full else 0.
        action = choose_action(target, self.rng, temperature)
        self.features.append(self.board.features().copy())
        self.mask.append(self.full); self.policies.append(target); self.values.append(value)
        self.visits.append(self.done_visits()); self.total_visits += self.done_visits()
        # Resignation: value is from the side to move's view.
        if self.allow_resign:
            c = 0 if self.board.to_move == Color.BLACK else 1
            self.low_streak[c] = self.low_streak[c] + 1 if value < -self.cfg.resign_threshold else 0
            if self.low_streak[c] >= self.cfg.resign_moves:
                self.resigned_winner = -1 if self.board.to_move == Color.BLACK else 1
                return
        idx = int(np.flatnonzero(root.moves == action)[0]) if action in root.moves else None
        self.board.play(action); self.moves.append(action)
        child = root.children.get(idx) if (self.cfg.reuse_tree and idx is not None) else None
        self.root = child if (child is not None and child.moves.size) else None
        if not self.finished:
            self._new_turn()

    def record(self):
        p = self.cfg.play
        return {'moves': np.asarray(self.moves, np.int16), 'policy': np.asarray(self.policies, np.float32),
                'features': np.asarray(self.features, np.uint8), 'train_mask': np.asarray(self.mask, np.bool_),
                'search_values': np.asarray(self.values, np.float32), 'visits': np.asarray(self.visits, np.int32),
                'board_size': p.board_size, 'komi': p.komi,
                'capped': not self.board.game_over and self.resigned_winner == 0,
                'resigned': self.resigned_winner != 0, 'resign_enabled': self.allow_resign,
                'judged': False, 'score': float('nan'),
                'ownership': np.zeros(p.board_size ** 2, np.int8), 'dead': np.empty(0, np.int16)}


class ParallelSelfPlay:
    """Drive many games; `evaluator(boards) -> (logits, values)` is called once per round."""

    def __init__(self, evaluator, cfg: ParallelConfig, seeds, on_finish, timers=None):
        self.evaluator, self.cfg, self.on_finish = evaluator, cfg, on_finish
        self.pending_seeds = list(seeds)
        self.games = []
        self.timers = timers if timers is not None else {}

    def _tick(self, key, t0):
        self.timers[key] = self.timers.get(key, 0.) + time.perf_counter() - t0

    def run(self):
        cfg = self.cfg
        while self.pending_seeds or self.games:
            while self.pending_seeds and len(self.games) < cfg.parallel_games:
                gid, seed = self.pending_seeds.pop(0)
                self.games.append(Game(gid, seed, cfg))
            t0 = time.perf_counter()
            requests = []  # (game, kind, payload)
            for g in self.games:
                if g.root is None:
                    requests.append((g, 'root', g.board))
                else:
                    want = min(cfg.leaves_per_game, g.target_visits - g.done_visits())
                    for path, b, cached in g.select_leaves(want):
                        requests.append((g, 'leaf', (path, b, cached)))
            self._tick('select', t0)
            boards = []
            for g, kind, payload in requests:
                b = payload if kind == 'root' else payload[1]
                if b is not None:
                    boards.append(b)
            t0 = time.perf_counter()
            nodes = []
            for i in range(0, len(boards), cfg.max_eval_batch):
                chunk = boards[i:i + cfg.max_eval_batch]
                logits, values = self.evaluator(chunk)
                nodes.extend(make_nodes(chunk, logits, values))
            self._tick('evaluate', t0)
            t0 = time.perf_counter()
            it = iter(nodes)
            for g, kind, payload in requests:
                if kind == 'root':
                    g.set_root(next(it))
                else:
                    path, b, cached = payload
                    g.backup(path, cached if b is None else next(it))
            still = []
            for g in self.games:
                if g.root is not None and g.done_visits() >= g.target_visits:
                    g.play_move()
                if g.finished:
                    self.on_finish(g)
                else:
                    still.append(g)
            self.games = still
            self._tick('backup_and_move', t0)
