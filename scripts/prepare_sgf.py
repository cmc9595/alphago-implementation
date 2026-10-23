"""Convert SGF archives into a compact game file for supervised training.

Only move sequences are stored (2 bytes per move); features are regenerated on the
fly by the C++ engine during training, so 40M positions take ~80MB instead of ~60GB.

    python scripts/prepare_sgf.py data/raw/kgs --out data/kgs.npz
Inputs may be .sgf files, directories, .zip, .tar(.gz|.bz2), or .7z (needs py7zr).
"""

import argparse
import hashlib
from array import array
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from alphago import Board
from alphago.sgf import parse_games


def iter_sgf_texts(path: Path):
    name = path.name.lower()
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.is_file():
                yield from iter_sgf_texts(p)
    elif name.endswith(".sgf"):
        yield path.read_bytes()
    elif name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.filename.lower().endswith(".sgf"):
                    yield z.read(info)
    elif name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        with tarfile.open(path, "r:*") as t:
            for m in t:
                if m.isfile() and m.name.lower().endswith(".sgf"):
                    yield t.extractfile(m).read()
    elif name.endswith(".7z") or ".7z.0" in name:
        import py7zr

        # Extract next to the archive once; reading from disk keeps memory flat.
        out = path.parent / "extracted" / path.name.split(".")[0]
        if not (out / ".done").exists():
            out.mkdir(parents=True, exist_ok=True)
            with py7zr.SevenZipFile(path) as z:
                z.extractall(path=out)
            (out / ".done").touch()
        yield from iter_sgf_texts(out)


def decode(raw: bytes) -> str:
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def validate(game, min_moves):
    """Return the legal action list, or a rejection reason."""
    if game.size != 19:
        return "not_19x19"
    if game.handicap > 0 or game.setup_black or game.setup_white:
        return "handicap"
    if len(game.moves) < min_moves:
        return "too_short"
    b = Board(19, game.komi if game.komi is not None else 7.5)
    actions = []
    for i, (color, action) in enumerate(game.moves):
        if color != (1 if i % 2 == 0 else 2):
            return "not_alternating"
        if not b.is_legal(action):
            return "illegal_move"
        b.play(action)
        actions.append(action)
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-moves", type=int, default=50)
    args = ap.parse_args()

    # Compact typed arrays: ~100M moves as Python ints would need several GB.
    moves, offsets, results, komis = array("h"), array("q", [0]), array("b"), array("f")
    stats, seen = Counter(), set()
    for inp in args.inputs:
        for raw in iter_sgf_texts(inp):
            try:
                games = parse_games(decode(raw))
            except Exception:
                stats["parse_error"] += 1
                continue
            for g in games:
                try:
                    v = validate(g, args.min_moves)
                except Exception:
                    v = "parse_error"
                if isinstance(v, str):
                    stats[v] += 1
                    continue
                digest = hashlib.blake2b(np.asarray(v, np.int16).tobytes(), digest_size=16).digest()
                if digest in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(digest)
                stats["kept"] += 1
                moves.extend(v)
                offsets.append(len(moves))
                results.append(g.result)
                komis.append(np.nan if g.komi is None else g.komi)
                if stats["kept"] % 20000 == 0:
                    print(f"  {dict(stats)}", file=sys.stderr, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        moves=np.frombuffer(moves, np.int16),
        offsets=np.frombuffer(offsets, np.int64),
        results=np.frombuffer(results, np.int8),
        komi=np.frombuffer(komis, np.float32),
    )
    print(f"games: {dict(stats)}")
    print(f"positions: {len(moves):,}  -> {args.out}")
    r = Counter(results)
    print(f"results: black {r[1]}, white {r[-1]}, unknown {r[0]}")


if __name__ == "__main__":
    main()
