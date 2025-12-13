from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Iterable
from .constants import Stone, opponent
from .zobrist import Zobrist

Coord = Tuple[int, int]  # (x, y), 0-based

@dataclass(frozen=True)
class Move:
    x: int
    y: int
    is_pass: bool = False
    is_resign: bool = False

    @staticmethod
    def pass_move():
        return Move(-1, -1, is_pass=True)

    @staticmethod
    def resign():
        return Move(-1, -1, is_resign=True)

class IllegalMove(Exception):
    pass

from go.zobrist import get_zobrist

class Board:
    def __init__(self, size: int = 19, komi: float = 7.5, superko: bool = False):
        self.size = size
        self.komi = komi
        self.superko = superko

        self.grid = [[Stone.EMPTY for _ in range(size)] for _ in range(size)]
        self.to_play = Stone.BLACK

        # self.zob = Zobrist(size=size, seed=0)
        self.zob = get_zobrist(size=size, seed=0)
        self.hash = self.zob.hash_empty() ^ self.zob.black_to_play  # black to play at start

        self.ko_point: Optional[Coord] = None  # simple ko
        self.history_hashes: Set[int] = {self.hash}

        self.captures_black = 0
        self.captures_white = 0
        self.pass_count = 0
        self.game_over = False
        self.winner: Optional[Stone] = None  # None until finished

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def neighbors(self, x: int, y: int) -> Iterable[Coord]:
        if x > 0: yield (x - 1, y)
        if x < self.size - 1: yield (x + 1, y)
        if y > 0: yield (x, y - 1)
        if y < self.size - 1: yield (x, y + 1)

    def get(self, x: int, y: int) -> Stone:
        return self.grid[y][x]

    def _set(self, x: int, y: int, c: Stone):
        prev = self.grid[y][x]
        if prev == c:
            return
        # update zobrist hash
        if prev in (Stone.BLACK, Stone.WHITE):
            self.hash ^= self.zob.table[x][y][prev]
        self.grid[y][x] = c
        if c in (Stone.BLACK, Stone.WHITE):
            self.hash ^= self.zob.table[x][y][c]

    def _toggle_player_hash(self):
        self.hash ^= self.zob.black_to_play

    def _collect_group(self, start: Coord) -> Tuple[Set[Coord], Set[Coord]]:
        """return (stones_in_group, liberties)"""
        sx, sy = start
        color = self.get(sx, sy)
        assert color != Stone.EMPTY

        group: Set[Coord] = set()
        libs: Set[Coord] = set()
        stack = [start]
        group.add(start)

        while stack:
            x, y = stack.pop()
            for nx, ny in self.neighbors(x, y):
                v = self.get(nx, ny)
                if v == Stone.EMPTY:
                    libs.add((nx, ny))
                elif v == color and (nx, ny) not in group:
                    group.add((nx, ny))
                    stack.append((nx, ny))
        return group, libs

    def _remove_group(self, group: Set[Coord]):
        for x, y in group:
            self._set(x, y, Stone.EMPTY)

    def is_legal(self, move: Move) -> bool:
        if self.game_over:
            return False
        if move.is_pass or move.is_resign:
            return True
        x, y = move.x, move.y
        if not self.in_bounds(x, y):
            return False
        if self.get(x, y) != Stone.EMPTY:
            return False
        if self.ko_point is not None and (x, y) == self.ko_point:
            return False

        # Try-play on a copy-lite: place stone, capture, check suicide, ko/superko hash.
        snapshot = self._snapshot_min()

        try:
            self._play_stone_no_checks(x, y)
            # suicide check: new group must have liberties after captures
            group, libs = self._collect_group((x, y))
            if len(libs) == 0:
                return False

            # superko check
            if self.superko and self.hash in self.history_hashes:
                return False
            return True
        finally:
            self._restore_min(snapshot)

    def play(self, move: Move):
        if not self.is_legal(move):
            raise IllegalMove(f"Illegal move: {move}")

        if move.is_resign:
            self.game_over = True
            self.winner = opponent(self.to_play)
            return

        if move.is_pass:
            self.pass_count += 1
            self.ko_point = None
            self._toggle_turn()
            if self.pass_count >= 2:
                self.game_over = True
            return

        self.pass_count = 0
        self.ko_point = None
        self._play_stone_no_checks(move.x, move.y)

        # After a real move, commit hash history
        self.history_hashes.add(self.hash)

        self._toggle_turn()

    def _toggle_turn(self):
        self.to_play = opponent(self.to_play)
        self._toggle_player_hash()

    def _play_stone_no_checks(self, x: int, y: int):
        color = self.to_play
        opp = opponent(color)

        self._set(x, y, color)

        captured_total = 0
        captured_single: Optional[Coord] = None

        # capture adjacent opponent groups with 0 liberties
        for nx, ny in self.neighbors(x, y):
            if self.get(nx, ny) != opp:
                continue
            group, libs = self._collect_group((nx, ny))
            if len(libs) == 0:
                captured_total += len(group)
                if len(group) == 1:
                    captured_single = next(iter(group))
                self._remove_group(group)

        # update capture counts
        if captured_total > 0:
            if color == Stone.BLACK:
                self.captures_black += captured_total
            else:
                self.captures_white += captured_total

        # simple ko: if exactly 1 stone captured and our placed stone group size==1 and created a ko shape
        if captured_total == 1:
            # if the placed stone is alone (group size 1) and opponent can recapture immediately -> mark ko point
            group, libs = self._collect_group((x, y))
            if len(group) == 1 and len(libs) == 1 and captured_single is not None:
                self.ko_point = captured_single

    def legal_moves(self) -> List[Move]:
        moves: List[Move] = [Move.pass_move(), Move.resign()]
        for y in range(self.size):
            for x in range(self.size):
                m = Move(x, y)
                if self.is_legal(m):
                    moves.append(m)
        return moves

    # ----- minimal snapshot/restore (for is_legal simulation) -----
    def _snapshot_min(self):
        return ( [row[:] for row in self.grid],
                 self.to_play, self.hash, self.ko_point,
                 self.captures_black, self.captures_white,
                 self.pass_count )

    def _restore_min(self, snap):
        (g, self.to_play, self.hash, self.ko_point,
         self.captures_black, self.captures_white,
         self.pass_count) = snap
        self.grid = g

    def copy(self) -> "Board":
        b = Board(size=self.size, komi=self.komi, superko=self.superko)

        # grid
        b.grid = [row[:] for row in self.grid]

        # game state
        b.to_play = self.to_play
        b.hash = self.hash
        b.ko_point = self.ko_point
        b.history_hashes = set(self.history_hashes)

        b.captures_black = self.captures_black
        b.captures_white = self.captures_white
        b.pass_count = self.pass_count
        b.game_over = self.game_over
        b.winner = self.winner

        return b

    # go/board.py (Board class 안에 추가)
    def place_setup_stone(self, x: int, y: int, stone: Stone):
        # SGF AB/AW용: 규칙 체크 없이 배치
        if stone == Stone.EMPTY:
            return
        if self.get(x, y) != Stone.EMPTY:
            return
        self.grid[y][x] = stone
