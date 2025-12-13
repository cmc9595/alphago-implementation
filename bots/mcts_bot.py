from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Callable

from bots.base import Bot
from go.board import Board, Move
from go.constants import Stone, opponent
from go.scoring import chinese_area_score

from typing import NamedTuple

class MoveStat(NamedTuple):
    move: Move
    visits: int
    q: float   # [-1,1] root perspective


def _result_value_from_root_perspective(
    board: Board,
    root_player: Stone,
) -> float:
    """
    Returns value in [-1, +1] from root_player's perspective.
    Uses:
      - resign winner if set
      - else Chinese area score
    """
    if board.winner is not None:
        return 1.0 if board.winner == root_player else -1.0

    b, w = chinese_area_score(board.grid, board.komi)
    winner = Stone.BLACK if b > w else Stone.WHITE
    return 1.0 if winner == root_player else -1.0


def _rollout(board: Board, root_player: Stone, rollout_depth: int) -> float:
    """
    Depth-limited random rollout.
    - Prefer non-resign moves
    - Stop if game_over (2 passes or resign)
    - If depth hits, evaluate by score.
    """
    for _ in range(rollout_depth):
        if board.game_over:
            break
        moves = board.legal_moves()
        # remove resign (초기엔 이상행동 방지)
        moves = [m for m in moves if not m.is_resign]
        if not moves:
            board.play(Move.pass_move())
            continue
        m = random.choice(moves)
        board.play(m)

    # finalize if not ended: evaluate current position by score anyway
    return _result_value_from_root_perspective(board, root_player)


@dataclass
class Node:
    parent: Optional["Node"]
    move: Optional[Move]
    player_to_play: Stone  # player to play at this node
    untried_moves: List[Move]
    children: Dict[Move, "Node"] = field(default_factory=dict)

    N: int = 0       # visit count
    W: float = 0.0   # total value (root perspective)
    Q: float = 0.0   # mean value (root perspective)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def best_child_uct(self, c_uct: float) -> "Node":
        """
        UCT: Q + c * sqrt(log(Nparent) / (1 + Nchild))
        """
        assert self.children, "No children to select"

        log_N = math.log(self.N + 1e-9)
        best_score = -1e18
        best = None
        for child in self.children.values():
            u = c_uct * math.sqrt(log_N / (child.N + 1))
            score = child.Q + u
            if score > best_score:
                best_score = score
                best = child
        return best


class MCTSBot(Bot):
    def __init__(
        self,
        num_simulations: int = 400,
        c_uct: float = 1.4,
        rollout_depth: int = 120,
        seed: Optional[int] = None,
    ):
        self.num_simulations = num_simulations
        self.c_uct = c_uct
        self.rollout_depth = rollout_depth
        if seed is not None:
            random.seed(seed)

    def select_move_with_stats(
        self,
        board: Board,
        topk: int = 5,
        progress_cb: Optional[Callable[[int, int, list[MoveStat]], None]] = None,
        progress_every: int = 25,
    ) -> tuple[Move, list[MoveStat]]:
        root_player = board.to_play
        root = self._make_node(board, parent=None, move=None)

        def current_topk() -> list[MoveStat]:
            if not root.children:
                return []
            items = list(root.children.items())
            items.sort(key=lambda kv: kv[1].N, reverse=True)
            return [MoveStat(m, child.N, child.Q) for m, child in items[:topk]]

        for t in range(1, self.num_simulations + 1):
            sim_board = board.copy()
            node = root

            while node.is_fully_expanded() and node.children:
                node = node.best_child_uct(self.c_uct)
                sim_board.play(node.move)  # type: ignore[arg-type]

            if node.untried_moves and not sim_board.game_over:
                m = node.untried_moves.pop()
                sim_board.play(m)
                child = self._make_node(sim_board, parent=node, move=m)
                node.children[m] = child
                node = child

            value = _rollout(sim_board, root_player, self.rollout_depth)
            self._backup(node, value)

            # ✅ 중간 진행 콜백
            if progress_cb is not None and (t % progress_every == 0 or t == self.num_simulations):
                progress_cb(t, self.num_simulations, current_topk())

        stats = current_topk()
        if not stats:
            return Move.pass_move(), []

        best_move = stats[0].move
        return best_move, stats



    def select_move(self, board: Board) -> Move:
        # Root player is the one who is to play at the root
        root_player = board.to_play

        root = self._make_node(board, parent=None, move=None)

        for _ in range(self.num_simulations):
            sim_board = board.copy()
            node = root

            # 1) Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child_uct(self.c_uct)
                sim_board.play(node.move)  # type: ignore[arg-type]

            # 2) Expansion
            if node.untried_moves and not sim_board.game_over:
                m = node.untried_moves.pop()
                sim_board.play(m)
                child = self._make_node(sim_board, parent=node, move=m)
                node.children[m] = child
                node = child

            # 3) Simulation
            value = _rollout(sim_board, root_player, self.rollout_depth)

            # 4) Backprop (root perspective value)
            self._backup(node, value)

        # choose move with max visits
        if not root.children:
            return Move.pass_move()

        best_move, best_child = max(root.children.items(), key=lambda kv: kv[1].N)
        return best_move

    def _make_node(self, board: Board, parent: Optional[Node], move: Optional[Move]) -> Node:
        legal = board.legal_moves()
        # remove resign for MCTS default
        legal = [m for m in legal if not m.is_resign]
        # shuffle so pop() gives random expansion order
        random.shuffle(legal)

        return Node(
            parent=parent,
            move=move,
            player_to_play=board.to_play,
            untried_moves=legal,
        )

    def _backup(self, node: Node, value: float):
        """
        value is always from ROOT player's perspective.
        So every node accumulates the same sign.
        """
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent
