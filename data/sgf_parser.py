# data/sgf_parser.py (parse_sgf_game 교체)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
from sgfmill import sgf, sgf_moves


@dataclass
class SgfGame:
    source: str
    size: int
    komi: float
    handicap: int
    setup_black: List[Tuple[int,int]]  # (x,y)
    setup_white: List[Tuple[int,int]]  # (x,y)
    moves: List[Tuple[str, Optional[Tuple[int,int]]]]  # ("b"/"w", (x,y)) or None pass

    # ✅ 추가: 랭크
    black_rank: str | None = None  # BR
    white_rank: str | None = None  # WR

def _safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _sgf_points_to_xy(points_rc, size: int) -> List[Tuple[int,int]]:
    """
    sgfmill point coords are (row, col)
    convert to (x=col, y=row)
    """
    out: List[Tuple[int,int]] = []
    if not points_rc:
        return out
    for rc in points_rc:
        if rc is None:
            continue
        r, c = rc
        if 0 <= r < size and 0 <= c < size:
            out.append((c, r))
    return out


def parse_sgf_game(source: str, data: bytes) -> Optional[SgfGame]:
    try:
        g = sgf.Sgf_game.from_bytes(data)
    except Exception:
        return None

    size = g.get_size()
    root = g.get_root()

    km_raw = root.get("KM") if root.has_property("KM") else 0.0
    ha_raw = root.get("HA") if root.has_property("HA") else 0

    komi = _safe_float(km_raw, 0.0)
    try:
        handicap = int(ha_raw or 0)
    except Exception:
        handicap = 0

    # ✅ 랭크 (예: "6d", "2k", "7d?", "6d*")
    br = root.get("BR") if root.has_property("BR") else None
    wr = root.get("WR") if root.has_property("WR") else None
    black_rank = str(br) if br is not None else None
    white_rank = str(wr) if wr is not None else None

    # ✅ AB/AW는 root에서 직접 읽는다 (가장 안정)
    ab_rc = root.get("AB") if root.has_property("AB") else []
    aw_rc = root.get("AW") if root.has_property("AW") else []

    setup_black = _sgf_points_to_xy(ab_rc, size)
    setup_white = _sgf_points_to_xy(aw_rc, size)

    try:
        _setup_board, plays = sgf_moves.get_setup_and_moves(g)
        # plays: list[(color, (row,col)|None)]
    except Exception:
        return None


    moves: List[Tuple[str, Optional[Tuple[int,int]]]] = []
    for color, mv in plays:
        if mv is None:
            moves.append((color, None))
        else:
            r, c = mv
            moves.append((color, (c, r)))

    return SgfGame(
        source=source,
        size=size,
        komi=komi,
        handicap=handicap,
        setup_black=setup_black,
        setup_white=setup_white,
        moves=moves,
        black_rank=br,
        white_rank=wr
    )

from pathlib import Path
from typing import Iterable, Iterator
import zipfile


def iter_sgf_bytes_from_zips(zip_paths: Iterable[Path]) -> Iterator[tuple[str, bytes]]:
    """
    Yields (source_id, sgf_bytes) from given zip files.
    source_id example: "KGS-2019_04-19-1255-.zip::foo.sgf"
    """
    for zpath in zip_paths:
        with zipfile.ZipFile(zpath, "r") as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".sgf"):
                    continue
                try:
                    data = zf.read(name)
                    yield f"{zpath.name}::{name}", data
                except Exception:
                    continue


def iter_games_from_kgs_zips(zip_dir: Path) -> Iterator[SgfGame]:
    zips = sorted(zip_dir.glob("KGS-*.zip"))
    # print("ZIP count:", len(zips))
    for source, data in iter_sgf_bytes_from_zips(zips):
        game = parse_sgf_game(source, data)
        if game is None:
            continue
        # if game.size == 19:
            # print("ONE GAME:", game.source, "moves=", len(game.moves), "setupB=", len(game.setup_black), "setupW=", len(game.setup_white))
            # return  # ✅ 딱 한 개만 보고 종료(임시)
        yield game
