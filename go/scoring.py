from collections import deque
from typing import Set, Tuple
from .constants import Stone

Coord = Tuple[int, int]

def chinese_area_score(grid, komi: float) -> tuple[float, float]:
    size = len(grid)
    def neighbors(x,y):
        if x>0: yield x-1,y
        if x<size-1: yield x+1,y
        if y>0: yield x,y-1
        if y<size-1: yield x,y+1

    black_stones = sum(1 for y in range(size) for x in range(size) if grid[y][x]==Stone.BLACK)
    white_stones = sum(1 for y in range(size) for x in range(size) if grid[y][x]==Stone.WHITE)

    seen: Set[Coord] = set()
    black_terr = 0
    white_terr = 0

    for y in range(size):
        for x in range(size):
            if grid[y][x] != Stone.EMPTY or (x,y) in seen:
                continue
            q = deque([(x,y)])
            comp = {(x,y)}
            seen.add((x,y))
            border_colors = set()
            while q:
                cx, cy = q.popleft()
                for nx, ny in neighbors(cx,cy):
                    v = grid[ny][nx]
                    if v == Stone.EMPTY and (nx,ny) not in seen:
                        seen.add((nx,ny))
                        comp.add((nx,ny))
                        q.append((nx,ny))
                    elif v != Stone.EMPTY:
                        border_colors.add(v)
            if border_colors == {Stone.BLACK}:
                black_terr += len(comp)
            elif border_colors == {Stone.WHITE}:
                white_terr += len(comp)

    black = black_stones + black_terr
    white = white_stones + white_terr + komi
    return float(black), float(white)
