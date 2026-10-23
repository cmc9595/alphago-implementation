"""PUCT Monte-Carlo tree search with batched leaf evaluation (AlphaGo-style, value net only).

Each simulation descends by  argmax_a  Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)),
expands the leaf with policy priors, evaluates it with the value net, and backs the value up.
`batch` leaves are collected per network call using virtual loss so the GPU sees real batches.
"""

import math
import sys
import time

import numpy as np
import torch

from alphago import Board, features_batch


class Node:
    __slots__ = ("moves", "P", "N", "W", "children", "value")

    def __init__(self, moves: np.ndarray, priors: np.ndarray, value: float):
        self.moves = moves  # candidate actions (legal, not own-eye, plus pass)
        self.P = priors
        self.N = np.zeros(len(moves), np.float32)
        self.W = np.zeros(len(moves), np.float32)  # from the view of the player to move here
        self.children: dict[int, "Node"] = {}
        self.value = value  # network value of this node, side-to-move view


class TreeBudget:
    """Incremental accounting of retained nodes/arrays, not process or GPU memory.

    Nodes are counted only when attached to the tree (batch duplicates do not count).
    Reserve one worst-case node per simulation before running a batch. Temporary
    feature tensors, Python allocator reserves and GPU inference are additional.
    """
    def __init__(self, max_nodes=200_000, max_bytes=2 * 1024**3):
        if max_nodes < 1 or max_bytes < 1:
            raise ValueError("tree limits must be positive")
        self.max_nodes, self.max_bytes = max_nodes, max_bytes
        self.nodes = self.bytes = 0
        self.limit_reason = None

    @staticmethod
    def node_bytes(node):
        seen = set()
        def array_bytes(array):
            if id(array) in seen:
                return 0
            seen.add(id(array))
            return sys.getsizeof(array) + (array_bytes(array.base) if array.base is not None else 0)
        return (sys.getsizeof(node) + sys.getsizeof(node.children) + sys.getsizeof(node.value) +
                sum(array_bytes(a) for a in (node.moves, node.P, node.N, node.W)))

    def add_root(self, root):
        if self.nodes:
            raise ValueError("use a fresh budget for a new tree")
        self.nodes = 1
        self.bytes = self.node_bytes(root)

    def attach(self, parent, index, child):
        before = sys.getsizeof(parent.children)
        parent.children[index] = child
        self.nodes += 1
        self.bytes += self.node_bytes(child) + sys.getsizeof(parent.children) - before + sys.getsizeof(index)

    def capacity(self, board_size, requested):
        # Includes maximum action arrays, headers and a conservative allowance
        # for dictionary growth, even when the parent dictionary resizes.
        reserve = 20 * (board_size * board_size + 1) + 32768
        slots = min(self.max_nodes - self.nodes, (self.max_bytes - self.bytes) // reserve)
        if slots <= 0:
            self.limit_reason = "nodes" if self.nodes >= self.max_nodes else "memory"
            return 0
        return min(requested, int(slots))


SYMMETRIES = {1: (0,), 2: (0, 2), 4: (0, 1, 2, 3), 8: tuple(range(8))}


def transform_tensor(x, symmetry):
    """The same D4 convention as alphago.data.transform, on the last two axes."""
    out = torch.rot90(x, symmetry % 4, dims=(-2, -1))
    return torch.flip(out, dims=(-1,)) if symmetry >= 4 else out


@torch.no_grad()
def average_value(net, x, symmetries=1, identity_value=None):
    """Average scalar values; the policy is left unchanged. Never changes the player's colour."""
    if symmetries not in SYMMETRIES:
        raise ValueError("value symmetries must be 1, 2, 4, or 8")
    out = []
    for s in SYMMETRIES[symmetries]:
        v = identity_value if s == 0 and identity_value is not None else net(transform_tensor(x, s))[1]
        out.append(v)
    return torch.stack(out).mean(dim=0)


class NetEvaluator:
    """Unchanged policy priors, optionally symmetry-averaged scalar values."""

    def __init__(self, policy_net, value_net=None, device=None, value_symmetries=1):
        if value_symmetries not in SYMMETRIES:
            raise ValueError("value symmetries must be 1, 2, 4, or 8")
        self.policy_net = policy_net.eval()
        self.value_net = (value_net or policy_net).eval()
        self.device = device or next(policy_net.parameters()).device
        self.value_symmetries = value_symmetries

    @torch.no_grad()
    def __call__(self, boards):
        x = torch.from_numpy(features_batch(boards)).to(self.device).float()
        logits, identity = self.policy_net(x)
        identity = identity if self.value_net is self.policy_net else None
        value = average_value(self.value_net, x, self.value_symmetries, identity)
        return logits.float().cpu().numpy(), value.float().cpu().numpy()


class MCTS:
    def __init__(self, evaluator, playouts=800, batch=16, c_puct=1.5, virtual_loss=1.0, fpu_reduction=0.2, time_limit=None):
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time limit must be positive")
        self.time_limit = time_limit
        self.evaluate = evaluator
        self.playouts = playouts
        self.batch = batch
        self.c_puct = c_puct
        self.vl = virtual_loss
        self.fpu = fpu_reduction

    def _expand(self, boards):
        logits, values = self.evaluate(boards)
        nodes = []
        for b, lg, v in zip(boards, logits, values):
            if b.game_over:
                # Training targets use dead-stone adjudication. Raw Tromp-Taylor
                # counting is NOT an exact terminal result for these targets.
                # Freeze the learned value here; never search beyond two passes.
                nodes.append(Node(np.empty(0, np.int64), np.empty(0, np.float32), float(v)))
                continue
            moves = np.flatnonzero(b.sensible_mask())
            p = lg[moves] - lg[moves].max()
            p = np.exp(p)
            nodes.append(Node(moves, p / p.sum(), float(v)))
        return nodes

    def _select(self, node: Node) -> int:
        total = node.N.sum()
        visited = node.N > 0
        # First-play urgency: unvisited moves start a bit below the parent's current estimate.
        parent_q = (node.W.sum() / total) if total > 0 else node.value
        q = np.where(visited, node.W / np.maximum(node.N, 1e-9), parent_q - self.fpu)
        u = self.c_puct * node.P * math.sqrt(total + 1) / (1 + node.N)
        return int(np.argmax(q + u))

    def search(self, board: Board, root: Node | None = None, stop_event=None, budget: TreeBudget | None = None) -> Node:
        if board.game_over:
            raise ValueError("cannot search a finished game")
        started = time.perf_counter()
        # A supplied root must belong to this exact position. Callers invalidate it on moves.
        if root is None:
            root = self._expand([board])[0]
            if budget is not None:
                budget.add_root(root)
        elif budget is not None and not budget.nodes:
            raise ValueError("budget must track this tree from its first expansion")
        done = 0
        while done < self.playouts:
            if stop_event is not None and stop_event.is_set():
                break
            if done and self.time_limit is not None and time.perf_counter() - started >= self.time_limit:
                break
            count = min(self.batch, self.playouts - done)
            if budget is not None:
                count = budget.capacity(board.size, count)
                if not count:
                    break
            pending = []  # path, new leaf board (or None), cached terminal node (or None)
            for _ in range(count):
                node, b, path = root, board.copy(), []
                while node.moves.size:
                    i = self._select(node)
                    path.append((node, i))
                    node.N[i] += self.vl
                    node.W[i] -= self.vl
                    b.play(int(node.moves[i]))
                    child = node.children.get(i)
                    if child is None:
                        pending.append((path, b, None))
                        break
                    node = child
                else:
                    # A cached terminal leaf: reuse its value, without playing or
                    # evaluating another move after the game has ended.
                    pending.append((path, None, node))
            boards = [b for _, b, _ in pending if b is not None]
            leaves = iter(self._expand(boards) if boards else [])
            for path, b, cached in pending:
                leaf = cached if cached is not None else next(leaves)
                parent, i = path[-1]
                if i not in parent.children:
                    if budget is None:
                        parent.children[i] = leaf
                    else:
                        budget.attach(parent, i, leaf)
                val = -leaf.value  # leaf side-to-move -> parent side-to-move
                for node, j in reversed(path):
                    node.N[j] += 1 - self.vl
                    node.W[j] += val + self.vl
                    val = -val
            done += len(pending)
        return root

    def genmove(self, board: Board) -> int:
        root = self.search(board)
        return int(root.moves[int(np.argmax(root.N))])

    @staticmethod
    def summary(root: Node, top=5):
        order = np.argsort(-root.N)[:top]
        return [{"action": int(root.moves[i]), "visits": int(root.N[i]),
                 "q": round(float(root.W[i] / max(root.N[i], 1)), 3), "prior": round(float(root.P[i]), 3)}
                for i in order]


class MCTSPlayer:
    """Same interface as gtp.PolicyPlayer (genmove / evaluate) so GTP, the web board and matches can use it."""

    def __init__(self, policy_net, value_net=None, playouts=800, batch=16, c_puct=1.5, value_symmetries=1, time_limit=None):
        self.net = policy_net
        self.device = next(policy_net.parameters()).device
        self.evaluator = NetEvaluator(policy_net, value_net, self.device, value_symmetries=value_symmetries)
        self.mcts = MCTS(self.evaluator, playouts=playouts, batch=batch, c_puct=c_puct, time_limit=time_limit)
        self.last_root = None

    def evaluate(self, board):
        logits, values = self.evaluator([board])
        lg = logits[0] - logits[0].max()
        p = np.exp(lg)
        return p / p.sum(), float(values[0])

    def genmove(self, board):
        self.last_root = self.mcts.search(board)
        return int(self.last_root.moves[int(np.argmax(self.last_root.N))])
