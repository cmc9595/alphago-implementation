# nn/features.py
from __future__ import annotations
import numpy as np
from go.board import Board, Move
from go.constants import Stone, opponent

def _one_hot_planes(val_grid: np.ndarray, max_k: int) -> np.ndarray:
    """
    val_grid: (19,19) int
    returns: (max_k,19,19) where plane i is (val==i+1), and last plane is (val>=max_k)
    """
    planes = np.zeros((max_k, val_grid.shape[0], val_grid.shape[1]), dtype=np.float32)
    for k in range(1, max_k):
        planes[k-1] = (val_grid == k).astype(np.float32)
    planes[max_k-1] = (val_grid >= max_k).astype(np.float32)
    return planes

def encode_policy_features(board: Board) -> np.ndarray:
    """
    returns (48,19,19) float32
    상대/자기 기준은 '현재 to_play' 기준으로 맞춤(논문 스타일).
    """
    n = board.size
    assert n == 19, "현재 MVP는 19x19 기준으로 먼저 고정"

    me = board.to_play
    opp = opponent(me)

    # 1) stone color 3
    player = np.zeros((n,n), np.float32)
    oppo   = np.zeros((n,n), np.float32)
    empty  = np.zeros((n,n), np.float32)
    for y in range(n):
        for x in range(n):
            s = board.get(x,y)
            if s == Stone.EMPTY:
                empty[y,x] = 1.0
            elif s == me:
                player[y,x] = 1.0
            else:
                oppo[y,x] = 1.0

    ones = np.ones((n,n), np.float32)
    zeros = np.zeros((n,n), np.float32)

    # 2) liberties(현재 돌들의 체인 liberty 수)
    # board 쪽에 helper가 없으면: 일단 간단 버전(EMPTY는 0)로 구현
    libs = np.zeros((n,n), np.int32)
    for y in range(n):
        for x in range(n):
            if board.get(x,y) == Stone.EMPTY:
                continue
            libs[y,x] = min(board.count_liberties(x,y), 8)  # 너 엔진에 함수 있다고 가정

    libs_planes = _one_hot_planes(libs, 8)

    # 3) sensibleness: legal and not self-eye fill (MVP: legal만)
    sensible = np.zeros((n,n), np.float32)
    for y in range(n):
        for x in range(n):
            if board.get(x,y) != Stone.EMPTY:
                continue
            if board.is_legal(Move(x,y)):
                sensible[y,x] = 1.0

    # 4) capture size / self-atari / liberties after move: “그 자리에 둘 때”를 전수 체크(비싸지만 MVP OK)
    cap = np.zeros((n,n), np.int32)
    self_atari = np.zeros((n,n), np.int32)
    libs_after = np.zeros((n,n), np.int32)
    for y in range(n):
        for x in range(n):
            if board.get(x,y) != Stone.EMPTY:
                continue
            m = Move(x,y)
            if not board.is_legal(m):
                continue
            b2 = board.copy()
            before_my_stones = b2.count_stones(me)
            before_opp_stones = b2.count_stones(opp)
            b2.play(m)
            after_my_stones = b2.count_stones(me)
            after_opp_stones = b2.count_stones(opp)

            cap[y,x] = min(max(0, before_opp_stones - after_opp_stones), 8)
            self_atari[y,x] = min(max(0, before_my_stones - after_my_stones), 8)
            libs_after[y,x] = min(b2.count_liberties(x,y), 8)

    cap_planes = _one_hot_planes(cap, 8)
    self_atari_planes = _one_hot_planes(self_atari, 8)
    libs_after_planes = _one_hot_planes(libs_after, 8)

    # 5) turns since: 엔진에 history 없으면 전부 0으로 두자(MVP)
    turns_planes = np.zeros((8,n,n), np.float32)

    # 6) ladder planes: MVP는 0
    ladder_cap = np.zeros((n,n), np.float32)
    ladder_esc = np.zeros((n,n), np.float32)

    planes = np.concatenate([
        player[None], oppo[None], empty[None],      # 3
        ones[None],                                  # 1 -> 4
        turns_planes,                                # 8 -> 12
        libs_planes,                                 # 8 -> 20
        cap_planes,                                  # 8 -> 28
        self_atari_planes,                            # 8 -> 36
        libs_after_planes,                             # 8 -> 44
        ladder_cap[None], ladder_esc[None],          # 2 -> 46
        sensible[None],                               # 1 -> 47
        zeros[None],                                  # 1 -> 48
    ], axis=0).astype(np.float32)

    assert planes.shape == (48,19,19)
    return planes

def encode_value_features(board: Board) -> np.ndarray:
    """
    value는 policy 48 + (현재 플레이어가 흑인지) 1 plane 추가. :contentReference[oaicite:8]{index=8}
    """
    p = encode_policy_features(board)
    n = board.size
    plane = np.ones((1,n,n), np.float32) if board.to_play == Stone.BLACK else np.zeros((1,n,n), np.float32)
    return np.concatenate([p, plane], axis=0).astype(np.float32)  # (49,19,19)
