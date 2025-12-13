from enum import IntEnum

class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

def opponent(c: Stone) -> Stone:
    if c == Stone.BLACK:
        return Stone.WHITE
    if c == Stone.WHITE:
        return Stone.BLACK
    return Stone.EMPTY
