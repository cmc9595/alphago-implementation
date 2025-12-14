# bots/puct_bot.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List, Tuple, NamedTuple

import torch
import numpy as np

from bots.base import Bot
from bots.nn.policy_net import PolicyNet

from go.board import Board, Move
from go.constants import Stone
from go.scoring import chinese_area_score

from features.alpha48 import FeatureState, extract_alpha48_planes


# -----------------------------
# UI용 통계
# -----------------------------
class MoveStat(NamedTuple):
    move: Move
    visits: int
    q: float   # [-1,1] root perspective
    p: float   # prior


# -----------------------------
# value helper (root perspective)
# -----------------------------
def _result_value_from_root_perspective(board: Board, root_player: Stone) -> float:
    if board.winner is not None:
        return 1.0 if board.winner == root_player else -1.0
    b, w = chinese_area_score(board.grid, board.komi)
    winner = Stone.BLACK if b > w else Stone.WHITE
    return 1.0 if winner == root_player else -1.0


def _random_rollout(board: Board, root_player: Stone, rollout_depth: int) -> float:
    for _ in range(rollout_depth):
        if board.game_over:
            break
        moves = board.legal_moves()
        moves = [m for m in moves if not m.is_resign]
        if not moves:
            board.play(Move.pass_move())
            continue
        board.play(random.choice(moves))
    return _result_value_from_root_perspective(board, root_player)


# -----------------------------
# Policy prior
# -----------------------------
class PolicyPrior:
    """
    SL PolicyNet을 이용해 현재 보드에서 legal move priors를 만든다.
    - input: alpha48 planes (48,19,19)
    - output: softmax over 361(+pass)
    """
    def __init__(self, weight_path: str | Path, device: str = "cpu"):
        self.device = device
        self.model = PolicyNet().to(device)
        sd = torch.load(str(weight_path), map_location=device)
        # sd가 state_dict인지 {"model":...}인지 둘 다 대응
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        self.model.load_state_dict(sd)
        self.model.eval()

    @torch.no_grad()
    def priors(
        self,
        board: Board,
        st: FeatureState,
        allow_pass: bool = True,
    ) -> Dict[Move, float]:
        x = extract_alpha48_planes(
            board,
            st,
            include_value_color_plane=False,
            enable_ladder=False,   # ✅ 무조건 False
        )

        x_t = torch.from_numpy(x).unsqueeze(0).to(self.device)                 # (1,48,19,19)

        logits = self.model(x_t)[0]
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()

        size = board.size
        point_n = size * size            # 361
        has_pass_head = probs.shape[0] >= (point_n + 1)  # 모델이 pass까지 출력하는지

        pri: Dict[Move, float] = {}
        for m in board.legal_moves():
            if m.is_resign:
                continue

            if m.is_pass:
                # ✅ 모델이 pass를 출력하지 않으면 pass는 prior에서 제외
                if (not allow_pass) or (not has_pass_head):
                    continue
                idx = point_n
            else:
                idx = m.y * size + m.x
                if idx < 0 or idx >= min(probs.shape[0], point_n):
                    # 이론상 여기 오면 안 됨(방어)
                    continue

            pri[m] = float(probs[idx])

        s = sum(pri.values())
        if s > 0:
            for k in list(pri.keys()):
                pri[k] /= s
        return pri


# -----------------------------
# PUCT Node
# -----------------------------
@dataclass
class Node:
    parent: Optional["Node"]
    move: Optional[Move]

    # priors for this node's children (legal moves only)
    priors: Dict[Move, float]
    untried_moves: List[Move]
    children: Dict[Move, "Node"] = field(default_factory=dict)

    N: int = 0
    W: float = 0.0
    Q: float = 0.0

    # this node's prior from parent
    P: float = 0.0

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def best_child_puct(self, c_puct: float) -> "Node":
        assert self.children, "No children"
        sqrtN = math.sqrt(self.N + 1e-9)

        best_score = -1e18
        best: Optional[Node] = None

        for m, child in self.children.items():
            # Q: root-perspective mean value
            q = child.Q
            # PUCT exploration term
            p = child.P
            u = c_puct * p * (sqrtN / (1.0 + child.N))
            score = q + u
            if score > best_score:
                best_score = score
                best = child

        assert best is not None
        return best


# -----------------------------
# PUCTBot
# -----------------------------
class PUCTBot(Bot):
    def __init__(
        self,
        policy_weight: str | Path,
        device: str = "cpu",
        num_simulations: int = 800,
        c_puct: float = 1.5,
        rollout_depth: int = 0,          # 0이면 rollout 없이 "현재 스코어 기반" value로 근사
        allow_pass: bool = True,
        seed: Optional[int] = None,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.rollout_depth = rollout_depth
        self.allow_pass = allow_pass
        if seed is not None:
            random.seed(seed)

        self.prior_net = PolicyPrior(policy_weight, device=device)

    def select_move(self, board: Board) -> Move:
        move, _stats = self.select_move_with_stats(board, topk=0)
        return move

    def select_move_with_stats(
        self,
        board: Board,
        st: FeatureState,
        topk: int = 5,
        progress_cb=None,   # ✅ 추가
    ) -> Tuple[Move, List[MoveStat]]:
        root_player = board.to_play

        # root state
        # st_root = FeatureState.new(board.size)
        st_root = st  # ✅ UI에서 넘겨준 FeatureState가 루트
        # 중요: 현재 FeatureState를 "현재 board"에 맞게 만들 방법이 없다면,
        #       SL 학습 데이터처럼 "처음부터 수순을 apply" 해야 정확해짐.
        #       여기서는 UI에서 플레이할 때 st를 같이 들고 다니는 구조를 추천.
        #
        # 일단은: FeatureState가 보드만으로도 충분히 계산되는 plane이 많아서
        #         기본값으로도 돌아가게 두되, 성능은 떨어질 수 있음.
        #
        # => 아래 주석대로 UI/게임 루프에서 st를 유지하도록 바꾸면 성능 훨씬 좋아짐.

        root = self._make_node(board, st_root, parent=None, move=None, prior_p=1.0)

        for i in range(self.num_simulations):
            sim_board = board.copy()
            sim_st = st_root.copy() if hasattr(st_root, "copy") else self._cheap_copy_feature_state(st_root)

            node = root

            # 1) Selection
            while node.is_fully_expanded() and node.children and (not sim_board.game_over):
                node = node.best_child_puct(self.c_puct)
                # apply selected move
                sim_st.apply_and_update(sim_board, node.move)  # type: ignore[arg-type]

            # 2) Expansion
            if (not sim_board.game_over) and node.untried_moves:
                # PUCT는 보통 prior 큰 것부터 확장하는 게 유리
                m = self._pop_best_prior_move(node)
                sim_st.apply_and_update(sim_board, m)

                child = self._make_node(sim_board, sim_st, parent=node, move=m, prior_p=node.priors.get(m, 0.0))
                node.children[m] = child
                node = child

            # 3) Evaluation (rollout or heuristic)
            if sim_board.game_over:
                value = _result_value_from_root_perspective(sim_board, root_player)
            else:
                if self.rollout_depth > 0:
                    b2 = sim_board.copy()
                    value = _random_rollout(b2, root_player, self.rollout_depth)
                else:
                    # rollout 없으면 "현재 스코어 승패"로 약식 value
                    value = _result_value_from_root_perspective(sim_board, root_player)

            # 4) Backup
            self._backup(node, value)

            if progress_cb and (i % 20 == 0):
                stats = self._collect_root_stats(root, topk)
                progress_cb(i, self.num_simulations, stats)

        if not root.children:
            return Move.pass_move(), []

        # best move: most visits
        children = list(root.children.items())
        children.sort(key=lambda kv: kv[1].N, reverse=True)
        best_move = children[0][0]

        stats: List[MoveStat] = []
        if topk and topk > 0:
            for m, child in children[:topk]:
                stats.append(MoveStat(move=m, visits=child.N, q=child.Q, p=child.P))

        return best_move, stats

    def _collect_root_stats(self, root: Node, topk: int):
        items = list(root.children.items())
        items.sort(key=lambda kv: kv[1].N, reverse=True)
        stats = []
        for m, child in items[:topk]:
            stats.append(MoveStat(
                move=m,
                visits=child.N,
                q=child.Q,
                p=child.P
            ))
        return stats

    # -----------------------------
    # internals
    # -----------------------------
    def _make_node(
        self,
        board: Board,
        st: FeatureState,
        parent: Optional[Node],
        move: Optional[Move],
        prior_p: float,
    ) -> Node:
        pri = self.prior_net.priors(board, st, allow_pass=self.allow_pass)
        moves = list(pri.keys())
        # expansion order: shuffle는 하고, pop_best_prior_move로 prior 큰 것부터 뽑는다
        random.shuffle(moves)

        return Node(
            parent=parent,
            move=move,
            priors=pri,
            untried_moves=moves,
            P=float(prior_p),
        )

    def _pop_best_prior_move(self, node: Node) -> Move:
        # untried_moves 중 prior 최대인 move 하나 꺼내기
        best_i = 0
        best_p = -1.0
        for i, m in enumerate(node.untried_moves):
            p = node.priors.get(m, 0.0)
            if p > best_p:
                best_p = p
                best_i = i
        return node.untried_moves.pop(best_i)

    def _backup(self, node: Node, value: float):
        while node is not None:
            node.N += 1
            node.W += value
            node.Q = node.W / node.N
            node = node.parent

    def _cheap_copy_feature_state(self, st: FeatureState) -> FeatureState:
        # FeatureState.copy()가 없을 때라도 최소한 동작하게 하는 fallback
        # (성능/정확도 떨어질 수 있음)
        ns = FeatureState.new(19)
        # 혹시 필요한 필드가 있으면 여기서 더 복사
        if hasattr(st, "ply"):
            ns.ply = st.ply
        if hasattr(st, "ages") and hasattr(ns, "ages"):
            ns.ages[...] = st.ages
        return ns
