from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np

from go.board import Board, Move, IllegalMove
from go.constants import Stone, opponent


# -------------------------
# Utils (board inspection)
# -------------------------
def neighbors4(x: int, y: int, size: int) -> Iterable[Tuple[int, int]]:
    if x > 0: yield (x - 1, y)
    if x < size - 1: yield (x + 1, y)
    if y > 0: yield (x, y - 1)
    if y < size - 1: yield (x, y + 1)


def get_grid(board: Board):
    """
    board.grid 접근이 가능하면 사용하고,
    아니면 board.get(x,y)로 읽어도 되게 fallback.
    """
    g = getattr(board, "grid", None)
    if g is not None:
        return g
    # fallback: build 2D
    size = board.size
    return [[board.get(x, y) for x in range(size)] for y in range(size)]


def chain_and_liberties(board: Board, sx: int, sy: int) -> Tuple[List[Tuple[int,int]], int]:
    """
    Returns (chain_points, liberty_count) for the stone at (sx,sy).
    """
    size = board.size
    color = board.get(sx, sy)
    if color == Stone.EMPTY:
        return ([], 0)

    stack = [(sx, sy)]
    visited = set([(sx, sy)])
    chain = []
    libs = set()

    while stack:
        x, y = stack.pop()
        chain.append((x, y))
        for nx, ny in neighbors4(x, y, size):
            s = board.get(nx, ny)
            if s == Stone.EMPTY:
                libs.add((nx, ny))
            elif s == color and (nx, ny) not in visited:
                visited.add((nx, ny))
                stack.append((nx, ny))

    return chain, len(libs)


def is_eye_like(board: Board, x: int, y: int, player: Stone) -> bool:
    """
    "자기 눈 메우기" 판정용 아주 기본적인 eye-like 체크.
    - 해당 점이 비어있고
    - 상하좌우가 전부 player(또는 보드 밖)로 둘러싸이면 eye-like로 봄
    (진짜 eye/false eye는 코너/대각선까지 봐야하지만,
     AlphaGo의 'does not fill its own eyes'는 보통 이 정도로도 충분히 동작)
    """
    if board.get(x, y) != Stone.EMPTY:
        return False

    size = board.size
    for nx, ny in neighbors4(x, y, size):
        if board.get(nx, ny) != player:
            return False
    return True


def bucket_1_to_8(n: int) -> int:
    """
    n을 one-hot 8 plane의 index로 변환:
    1..7 -> 0..6
    >=8  -> 7
    0 or negative -> -1 (no plane)
    """
    if n <= 0:
        return -1
    if n >= 8:
        return 7
    return n - 1


def simulate_play(board: Board, move: Move) -> Optional[Board]:
    """
    Illegal이면 None, 아니면 play된 board copy 반환
    """
    b = board.copy()
    try:
        b.play(move)
        return b
    except IllegalMove:
        return None


def capture_size_if_play(board: Board, move: Move) -> int:
    """
    "이 점에 두면 즉시 따내는 상대 돌 수"
    """
    b2 = simulate_play(board, move)
    if b2 is None:
        return 0
    # before/after 비교로 잡힌 돌 수 계산 (간단/확실)
    size = board.size
    cnt = 0
    for y in range(size):
        for x in range(size):
            if board.get(x, y) == opponent(board.to_play) and b2.get(x, y) == Stone.EMPTY:
                cnt += 1
    return cnt


def self_atari_size_if_play(board: Board, move: Move) -> int:
    """
    "자충(1 liberty) 되는 내 체인 크기"를 self-atari size로 사용.
    (suicide는 일반 룰에서 illegal이라 보통 여기로 안 들어옴)
    """
    b2 = simulate_play(board, move)
    if b2 is None:
        return 0
    if move.is_pass or move.is_resign:
        return 0
    chain, libs = chain_and_liberties(b2, move.x, move.y)
    if libs == 1:
        return len(chain)
    return 0


def liberties_after_if_play(board: Board, move: Move) -> int:
    b2 = simulate_play(board, move)
    if b2 is None:
        return 0
    if move.is_pass or move.is_resign:
        return 0
    _, libs = chain_and_liberties(b2, move.x, move.y)
    return libs


# -------------------------
# Ladder (pragmatic, depth-limited)
# -------------------------
def ladder_capture_success(board: Board, move: Move, max_depth: int = 50) -> bool:
    """
    근사 ladder capture:
    - move를 두어서 상대 체인을 atari(1 liberty)로 만들었을 때,
    - defender가 가능한 최선의 탈출을 해도 결국 잡히는지 depth-limited로 읽음.
    """
    b0 = simulate_play(board, move)
    if b0 is None or move.is_pass or move.is_resign:
        return False

    attacker = board.to_play
    defender = opponent(attacker)

    # 방금 둔 수로 인해 atari 된 상대 체인 후보 찾기
    size = b0.size
    targets = []
    for nx, ny in neighbors4(move.x, move.y, size):
        if b0.get(nx, ny) == defender:
            ch, libs = chain_and_liberties(b0, nx, ny)
            if libs == 1:
                targets.append((ch, (nx, ny)))

    if not targets:
        return False

    # 어떤 target이라도 확정 ladder면 True로
    for _, anchor in targets:
        if _ladder_read(b0, anchor[0], anchor[1], attacker, defender, max_depth):
            return True
    return False


def _ladder_read(board: Board, sx: int, sy: int, attacker: Stone, defender: Stone, depth: int) -> bool:
    """
    간단 ladder minimax:
    - attacker 차례: atari 유지/강화하는 수가 있으면 계속
    - defender 차례: 탈출 가능한 수가 하나라도 있으면 실패
    """
    if depth <= 0:
        return False

    # 대상 체인 상태
    if board.get(sx, sy) != defender:
        # 이미 잡힘
        return True

    chain, libs = chain_and_liberties(board, sx, sy)
    if libs >= 3:
        return False
    if libs == 0:
        return True

    # defender to play?
    if board.to_play == defender:
        # defender가 살 수 있는 응수가 하나라도 있으면 ladder 실패
        # 1) atari 체인 liberty에 두기
        # 2) 상대 돌 따내서 liberty 늘리기
        legal = board.legal_moves()
        cand = []
        for m in legal:
            if m.is_resign:
                continue
            b2 = simulate_play(board, m)
            if b2 is None:
                continue
            # 대상 체인이 살아 있고 liberties가 2 이상이면 “탈출”로 판정
            if b2.get(sx, sy) == defender:
                _, libs2 = chain_and_liberties(b2, sx, sy)
                if libs2 >= 2:
                    return False
            # 아니면 계속 압박당하는 변형도 있으니 후보로만 추가
            cand.append(b2)

        # 모든 응수가 결국 잡히면 성공
        for b2 in cand:
            if not _ladder_read(b2, sx, sy, attacker, defender, depth - 1):
                return False
        return True

    # attacker to play
    else:
        # attacker는 “대상 체인 liberty를 줄이는 수”가 있으면 그 중 하나라도 성공이면 성공
        legal = board.legal_moves()
        best_moves = []
        for m in legal:
            if m.is_resign:
                continue
            b2 = simulate_play(board, m)
            if b2 is None:
                continue
            if b2.get(sx, sy) != defender:
                # 즉시 잡힘
                return True
            _, libs2 = chain_and_liberties(b2, sx, sy)
            if libs2 == 1:
                best_moves.append(b2)

        if not best_moves:
            return False

        for b2 in best_moves:
            if _ladder_read(b2, sx, sy, attacker, defender, depth - 1):
                return True
        return False


def ladder_escape_success(board: Board, move: Move, max_depth: int = 50) -> bool:
    """
    근사 ladder escape:
    - 현재 내 돌(체인)이 atari(1 liberty) 상태인 지점이 있고
    - move를 둔 뒤 그 체인이 2 liberties 이상으로 벗어나며
    - 이후 attacker가 ladder로 다시 잡지 못하면 escape 성공
    """
    player = board.to_play
    b0 = simulate_play(board, move)
    if b0 is None or move.is_pass or move.is_resign:
        return False

    # move 후, 주변 내 체인 중 "atari에서 벗어난 체인"이 있는지
    size = b0.size
    for nx, ny in neighbors4(move.x, move.y, size):
        if b0.get(nx, ny) == player:
            ch, libs = chain_and_liberties(b0, nx, ny)
            if libs >= 2:
                # 상대가 다시 ladder capture 가능한지(대충) 체크: 상대가 어떤 수로든 atari 만들고 ladder로 잡는지
                opp = opponent(player)
                # 상대 후보 수들 중 ladder capture 되는 수가 있으면 escape 실패로 봄(보수적)
                for m in b0.legal_moves():
                    if m.is_resign:
                        continue
                    if b0.to_play != opp:
                        break
                    # b0는 이미 move 후라서 to_play는 상대 차례일 것
                    if ladder_capture_success(b0, m, max_depth=max_depth):
                        return False
                return True
    return False

# -------------------------
# FeatureState (turn-since + recent moves)
# -------------------------
@dataclass
class FeatureState:
    """
    AlphaGo-style feature state.
    """
    size: int
    ages: np.ndarray                 # (size, size), int32, ply when stone placed
    last_moves: np.ndarray           # (8, size, size), float32
    ply: int = 0
    captures_black: int = 0
    captures_white: int = 0

    @classmethod
    def new(cls, size: int) -> "FeatureState":
        return cls(
            size=size,
            ages=np.full((size, size), -1, dtype=np.int32),
            last_moves=np.zeros((8, size, size), dtype=np.float32),
            ply=0,
            captures_black=0,
            captures_white=0,
        )

    def copy(self) -> "FeatureState":
        ns = FeatureState.new(self.size)
        ns.ply = self.ply
        ns.ages[...] = self.ages
        ns.last_moves[...] = self.last_moves
        ns.captures_black = self.captures_black
        ns.captures_white = self.captures_white
        return ns

    def apply_and_update(self, board: Board, move: Move):
        """
        board.play(move) 전후로 feature state 갱신
        """
        before = np.array(get_grid(board), dtype=np.int8)

        board.play(move)

        after = np.array(get_grid(board), dtype=np.int8)

        self.ply += 1

        # --- update ages
        placed = (before == Stone.EMPTY) & (after != Stone.EMPTY)
        removed = (before != Stone.EMPTY) & (after == Stone.EMPTY)

        self.ages[placed] = self.ply
        self.ages[removed] = -1

        # --- update captures (UI / stats 용)
        if board.to_play == Stone.WHITE:
            # 방금 Black이 둔 수
            self.captures_black = board.captures_black
        else:
            self.captures_white = board.captures_white

        # --- update last_moves (shift + one-hot)
        self.last_moves[1:] = self.last_moves[:-1]
        self.last_moves[0].fill(0.0)

        if not move.is_pass and not move.is_resign:
            self.last_moves[0, move.y, move.x] = 1.0




# -------------------------
# Main extractor
# -------------------------
def extract_alpha48_planes(board: Board, st: FeatureState, include_value_color_plane: bool = False, enable_ladder: bool = True)  -> np.ndarray:
    """
    Returns:
      - policy: (48, size, size) float32
      - value:  (49, size, size) float32 if include_value_color_plane
    All features are "relative to current player to play" (player/opponent) :contentReference[oaicite:3]{index=3}
    """
    size = board.size
    player = board.to_play
    opp = opponent(player)

    planes = []

    # --- Stone colour: 3
    player_st = np.zeros((size, size), np.float32)
    opp_st = np.zeros((size, size), np.float32)
    empty = np.zeros((size, size), np.float32)
    for y in range(size):
        for x in range(size):
            s = board.get(x, y)
            if s == player:
                player_st[y, x] = 1.0
            elif s == opp:
                opp_st[y, x] = 1.0
            else:
                empty[y, x] = 1.0
    planes += [player_st, opp_st, empty]

    # --- Ones: 1
    ones = np.ones((size, size), np.float32)
    planes.append(ones)

    # --- Turns since: 8 (for occupied points)
    turns_planes = [np.zeros((size, size), np.float32) for _ in range(8)]
    for y in range(size):
        for x in range(size):
            if board.get(x, y) == Stone.EMPTY:
                continue
            age = st.ages[y, x]
            if age < 0:
                continue
            turns_since = st.ply - int(age)
            idx = bucket_1_to_8(turns_since)
            if idx >= 0:
                turns_planes[idx][y, x] = 1.0
    planes += turns_planes

    # --- Liberties: 8 (for occupied points)
    libs_planes = [np.zeros((size, size), np.float32) for _ in range(8)]
    # compute per-chain once
    seen = set()
    for y in range(size):
        for x in range(size):
            if board.get(x, y) == Stone.EMPTY or (x, y) in seen:
                continue
            ch, libs = chain_and_liberties(board, x, y)
            for pt in ch:
                seen.add(pt)
            idx = bucket_1_to_8(libs)
            if idx >= 0:
                for cx, cy in ch:
                    libs_planes[idx][cy, cx] = 1.0
    planes += libs_planes

    # --- Capture size: 8 (for empty points; "if play here")
    cap_planes = [np.zeros((size, size), np.float32) for _ in range(8)]
    for y in range(size):
        for x in range(size):
            if board.get(x, y) != Stone.EMPTY:
                continue
            m = Move(x, y)
            if not board.is_legal(m):
                continue
            cap = capture_size_if_play(board, m)
            idx = bucket_1_to_8(cap)
            if idx >= 0:
                cap_planes[idx][y, x] = 1.0
    planes += cap_planes

    # --- Self-atari size: 8 (for empty points)
    selfa_planes = [np.zeros((size, size), np.float32) for _ in range(8)]
    for y in range(size):
        for x in range(size):
            if board.get(x, y) != Stone.EMPTY:
                continue
            m = Move(x, y)
            if not board.is_legal(m):
                continue
            sa = self_atari_size_if_play(board, m)
            idx = bucket_1_to_8(sa)
            if idx >= 0:
                selfa_planes[idx][y, x] = 1.0
    planes += selfa_planes

    # --- Liberties after move: 8 (for empty points)
    libs_after_planes = [np.zeros((size, size), np.float32) for _ in range(8)]
    for y in range(size):
        for x in range(size):
            if board.get(x, y) != Stone.EMPTY:
                continue
            m = Move(x, y)
            if not board.is_legal(m):
                continue
            la = liberties_after_if_play(board, m)
            idx = bucket_1_to_8(la)
            if idx >= 0:
                libs_after_planes[idx][y, x] = 1.0
    planes += libs_after_planes

    if enable_ladder:
        # ladder_cap / ladder_esc 계산
        # --- Ladder capture: 1 (for empty points)
        ladder_cap = np.zeros((size, size), np.float32)
        for y in range(size):
            for x in range(size):
                if board.get(x, y) != Stone.EMPTY:
                    continue
                m = Move(x, y)
                if not board.is_legal(m):
                    continue
                if ladder_capture_success(board, m, max_depth=50):
                    ladder_cap[y, x] = 1.0

        # --- Ladder escape: 1
        ladder_esc = np.zeros((size, size), np.float32)
        for y in range(size):
            for x in range(size):
                if board.get(x, y) != Stone.EMPTY:
                    continue
                m = Move(x, y)
                if not board.is_legal(m):
                    continue
                if ladder_escape_success(board, m, max_depth=50):
                    ladder_esc[y, x] = 1.0
    else:
        ladder_cap = np.zeros((size, size), np.float32)
        ladder_esc = np.zeros((size, size), np.float32)

    planes.append(ladder_cap)
    planes.append(ladder_esc)

    # --- Sensibleness: 1 (legal and does not fill its own eyes) :contentReference[oaicite:4]{index=4}
    sensible = np.zeros((size, size), np.float32)
    for y in range(size):
        for x in range(size):
            if board.get(x, y) != Stone.EMPTY:
                continue
            m = Move(x, y)
            if not board.is_legal(m):
                continue
            # "does not fill its own eyes" (근사)
            if is_eye_like(board, x, y, player) and capture_size_if_play(board, m) == 0:
                continue
            sensible[y, x] = 1.0
    planes.append(sensible)

    # --- Zeros: 1
    zeros = np.zeros((size, size), np.float32)
    planes.append(zeros)

    x = np.stack(planes, axis=0).astype(np.float32)
    assert x.shape[0] == 48, f"expected 48 planes, got {x.shape[0]}"

    # --- Value net extra plane: "player color (black?)" :contentReference[oaicite:5]{index=5}
    if include_value_color_plane:
        color_plane = np.ones((size, size), np.float32) if player == Stone.BLACK else np.zeros((size, size), np.float32)
        x = np.concatenate([x, color_plane[None, :, :]], axis=0)
        assert x.shape[0] == 49

    return x


