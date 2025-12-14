# data/sgf_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple
import re

import numpy as np

from go.board import Board, Move, IllegalMove
from go.constants import Stone

from data.sgf_parser import iter_games_from_kgs_zips, SgfGame
from features.alpha48 import FeatureState, extract_alpha48_planes

def _rank_to_dan(rank: str | None) -> int | None:
    """
    "6d", "7d?", "6d*", "2k" 등을 파싱해서
    dan이면 양의 정수, kyu면 음수로 반환.
    """
    if not rank:
        return None
    s = rank.strip().lower()
    m = re.search(r"(\d+)\s*([dk])", s)
    if not m:
        return None
    n = int(m.group(1))
    t = m.group(2)
    return n if t == "d" else -n


def _both_min_dan(game: SgfGame, min_dan: int) -> bool:
    bd = _rank_to_dan(game.black_rank)
    wd = _rank_to_dan(game.white_rank)
    return (bd is not None and wd is not None and bd >= min_dan and wd >= min_dan)

def move_to_index(move: Move, size: int = 19) -> int:
    # 0..360 for board points, 361 for pass (optional)
    if move.is_pass:
        return size * size
    if move.is_resign:
        return size * size + 1
    return move.y * size + move.x


@dataclass
class Sample:
    x: np.ndarray   # (48,19,19) float32
    y: int          # 0..360 (or 361 pass)
    source: str


def _color_to_stone(color: str) -> Stone:
    return Stone.BLACK if color.lower() == "b" else Stone.WHITE


def iter_sl_samples(
    zip_dir: Path,
    board_size: int = 19,
    include_pass: bool = False,
    max_games: Optional[int] = None,
    min_dan_both: int | None = None,   # ✅ 추가
    skip_handicap: bool = True,   # ✅ 추가
) -> Iterator[Sample]:
    """
    Streaming generator:
      yield Sample(x,y,source)
    """
    n_games = 0
    for game in iter_games_from_kgs_zips(zip_dir):
        if game.size != board_size:
            continue

        # ✅ 여기서 필터
        if min_dan_both is not None:
            if not _both_min_dan(game, min_dan_both):
                continue

        # ✅ handicap / AB/AW setup 스킵 (일단 파이프라인 살리기)
        if skip_handicap and (game.setup_black or game.setup_white):
            continue


        board = Board(size=board_size, komi=game.komi, superko=False)
        st = FeatureState.new(board_size)

        # --- apply handicap setup AB/AW
        black_xy, white_xy = game.setup_black, game.setup_white

        # 엔진이 set_stone 같은 게 있으면 그걸 쓰고,
        # 없으면 Board 내부 grid를 직접 건드리는 방법이 필요함.
        # 여기서는 "Board.place_setup_stone" 같은 메서드가 있다고 가정하고 없으면 아래 주석대로 추가해야 함.
        for (x, y) in black_xy:
            board.place_setup_stone(x, y, Stone.BLACK)  # 너 엔진에 추가 필요할 수 있음
            st.ages[y, x] = st.ply  # ply=0 시점에 배치된 것으로
        for (x, y) in white_xy:
            board.place_setup_stone(x, y, Stone.WHITE)
            st.ages[y, x] = st.ply

        # ✅ 여기! setup 직후에 to_play를 SGF 첫 수에 맞춤
        if game.moves:
            first_color = _color_to_stone(game.moves[0][0])
            board.to_play = first_color

        # --- play moves
        ok = True
        for color, mv in game.moves:
            # SGF color와 엔진 to_play가 어긋나면, SGF가 신뢰할 수 없는 게임이거나 variation/handicap 이슈
            # (handicap games는 종종 black 여러 수 연속이 아니라, setup 후 to_play가 white인 경우가 있음)
            want = _color_to_stone(color)
            if board.to_play != want:
                # SGFmill은 대체로 맞는데, 혹시 어긋나면 스킵(혹은 보정)
                ok = False
                break

            # next move parse
            if mv is None:
                move = Move.pass_move()
            else:
                x, y = mv
                move = Move(x, y)

            if move.is_pass and not include_pass:
                # pass를 학습 타겟에서 제외하고, 게임 진행만 반영할 수도 있음
                try:
                    st.apply_and_update(board, move)
                except IllegalMove:
                    ok = False
                    break
                continue

            # --- make sample BEFORE applying move
            x_planes = extract_alpha48_planes(board, st, include_value_color_plane=False, enable_ladder=False)

            y_idx = move_to_index(move, board_size)
            yield Sample(x=x_planes, y=y_idx, source=game.source)

            # --- apply move
            try:
                st.apply_and_update(board, move)
            except IllegalMove:
                ok = False
                break

            if board.game_over:
                break

        if not ok:
            continue

        n_games += 1
        if max_games is not None and n_games >= max_games:
            break
