"""Minimal SGF reader: main line only, which is all supervised training needs."""

from dataclasses import dataclass, field


@dataclass
class Game:
    size: int = 19
    komi: float | None = None
    handicap: int = 0
    result: int = 0  # +1 black won, -1 white won, 0 unknown / draw
    result_str: str = ""
    black_rank: str = ""
    white_rank: str = ""
    setup_black: list[int] = field(default_factory=list)
    setup_white: list[int] = field(default_factory=list)
    moves: list[tuple[int, int]] = field(default_factory=list)  # (color 1=B 2=W, action)


def _parse_nodes(text: str, start: int) -> tuple[list[dict[str, list[str]]], int]:
    """Parse one game tree starting at '(' and return its main-line nodes and end index.

    The main line is every node seen before the first ')': that paren closes the
    deepest first-variation branch, and everything after it is an alternative line.
    """
    nodes: list[dict[str, list[str]]] = []
    i, n, depth = start, len(text), 0
    done = False
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
            i += 1
        elif ch == ")":
            depth -= 1
            i += 1
            done = True
            if depth == 0:
                return nodes, i
        elif ch == ";":
            i += 1
            if not done:
                nodes.append({})
        elif ch.isupper():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            key = "".join(c for c in text[i:j] if c.isupper())  # FF4 drops lowercase letters
            i = j
            values = []
            while True:
                while i < n and text[i].isspace():
                    i += 1
                if i >= n or text[i] != "[":
                    break
                i += 1
                buf = []
                while i < n and text[i] != "]":
                    if text[i] == "\\" and i + 1 < n:
                        i += 1
                    buf.append(text[i])
                    i += 1
                i += 1
                values.append("".join(buf))
            if not done and nodes:
                nodes[-1].setdefault(key, []).extend(values)
        else:
            i += 1
    return nodes, n


def _point(value: str, size: int) -> int:
    """SGF 'pd' -> action index (y=0 is the bottom row); '' or 'tt' -> pass."""
    if value == "" or (value == "tt" and size <= 19):
        return size * size
    x = ord(value[0]) - ord("a")
    row = ord(value[1]) - ord("a")
    if not (0 <= x < size and 0 <= row < size):
        raise ValueError(f"bad point {value!r}")
    return (size - 1 - row) * size + x


def _result(value: str) -> int:
    v = value.strip().upper()
    if v.startswith("B+"):
        return 1
    if v.startswith("W+"):
        return -1
    return 0


def parse_games(text: str) -> list[Game]:
    """Parse every game tree in an SGF collection."""
    games = []
    i = text.find("(")
    while i != -1:
        nodes, end = _parse_nodes(text, i)
        if nodes:
            games.append(_to_game(nodes))
        i = text.find("(", end)
    return games


def _to_game(nodes: list[dict[str, list[str]]]) -> Game:
    root = nodes[0]
    g = Game()
    if "SZ" in root:
        g.size = int(root["SZ"][0].split(":")[0])
    if "KM" in root:
        try:
            g.komi = float(root["KM"][0])
        except ValueError:
            pass
    if "HA" in root:
        try:
            g.handicap = int(root["HA"][0])
        except ValueError:
            pass
    if "RE" in root:
        g.result_str = root["RE"][0]
        g.result = _result(g.result_str)
    g.black_rank = root.get("BR", [""])[0]
    g.white_rank = root.get("WR", [""])[0]
    for node in nodes:
        g.setup_black += [_point(v, g.size) for v in node.get("AB", [])]
        g.setup_white += [_point(v, g.size) for v in node.get("AW", [])]
        for key, color in (("B", 1), ("W", 2)):
            if key in node:
                g.moves.append((color, _point(node[key][0], g.size)))
    return g
