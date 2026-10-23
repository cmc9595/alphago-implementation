"""GNU Go over GTP: an opponent for matches and, more importantly, a dead-stone referee.

GNU Go reads each group's life and death explicitly (its "owl" module), so its final_score
and final_status_list are far more reliable than playing positions out with our policy.
"""

import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from alphago import Board

COLS = "ABCDEFGHJKLMNOPQRST"


def vertex(a: int, n: int = 19) -> str:
    return "pass" if a == n * n else f"{COLS[a % n]}{a // n + 1}"


def action(v: str, n: int = 19) -> int:
    v = v.strip().upper()
    return n * n if v == "PASS" else (int(v[1:]) - 1) * n + COLS.index(v[0])


def parse_score(s: str) -> float:
    """'B+7.5' -> 7.5, 'W+3.5' -> -3.5, '0' -> 0 (black minus white, komi included)."""
    s = s.strip().upper()
    if s.startswith("B+"):
        return float(s[2:])
    if s.startswith("W+"):
        return -float(s[2:])
    return 0.0


class GnuGo:
    def __init__(self, level=10, komi=7.5, superko=False):
        args = ["gnugo", "--mode", "gtp", "--level", str(level), "--chinese-rules", "--komi", str(komi)]
        if superko:
            args.append("--positional-superko")
        self.p = subprocess.Popen(
            args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )

    def cmd(self, c):
        self.p.stdin.write(c + "\n")
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("gnugo exited")
            if line.strip() == "" and lines:
                break
            if line.strip():
                lines.append(line.strip())
        resp = " ".join(lines)
        if not resp.startswith("="):
            raise RuntimeError(f"gnugo: {c} -> {resp}")
        return resp[1:].strip()

    def close(self):
        try:
            self.cmd("quit")
        except Exception:
            pass
        self.p.kill()


def ownership(moves, dead, n=19, komi=7.5) -> np.ndarray:
    """Final ownership per point (+1 Black, -1 White, 0 neutral) after removing dead stones."""
    b = Board(n, komi)
    for a in moves:
        b.play(a)
    stones = b.stones().ravel().astype(np.int8)  # 0 empty, 1 black, 2 white
    for a in dead:
        stones[a] = 0
    own = np.where(stones == 1, 1, np.where(stones == 2, -1, 0)).astype(np.int8)
    seen = np.zeros(n * n, bool)
    for start in np.flatnonzero(stones == 0):
        if seen[start]:
            continue
        region, borders, stack = [], set(), [start]
        seen[start] = True
        while stack:
            p = stack.pop()
            region.append(p)
            y, x = divmod(p, n)
            for q in ((p - n) if y > 0 else -1, (p + n) if y < n - 1 else -1,
                      (p - 1) if x > 0 else -1, (p + 1) if x < n - 1 else -1):
                if q < 0:
                    continue
                if stones[q] == 0:
                    if not seen[q]:
                        seen[q] = True
                        stack.append(q)
                else:
                    borders.add(int(stones[q]))
        if borders == {1}:
            own[region] = 1
        elif borders == {2}:
            own[region] = -1
    return own


class Referee:
    """A pool of GNU Go processes that score finished games in parallel."""

    def __init__(self, procs=8, level=5, komi=7.5):
        self.komi = komi
        self.level = level
        # A blocking queue: concurrent judge_many calls (e.g. training batch + evaluation) wait for
        # a free GNU Go process instead of failing.
        self.pool: queue.Queue = queue.Queue()
        self.procs = [GnuGo(level, komi) for _ in range(procs)]
        for g in self.procs:
            self.pool.put(g)
        self.ex = ThreadPoolExecutor(procs)

    def _judge(self, moves):
        g = self.pool.get()
        try:
            g.cmd("clear_board")
            for i, a in enumerate(moves):
                g.cmd(f"play {'B' if i % 2 == 0 else 'W'} {vertex(a)}")
            score = parse_score(g.cmd("final_score"))
            dead_s = g.cmd("final_status_list dead")
            dead = [action(v) for v in dead_s.split()] if dead_s else []
            return score, dead
        except Exception:
            g.close()  # a confused or dead process is replaced, never reused
            g = GnuGo(self.level, self.komi)
            self.procs.append(g)
            raise
        finally:
            self.pool.put(g)

    def judge_many(self, games):
        """games: list of move lists (Black first). Returns [(score, dead_actions) or None on failure]."""
        def safe(m):
            try:
                return self._judge(m)
            except Exception:
                return None
        return list(self.ex.map(safe, games))

    def close(self):
        self.ex.shutdown()
        for g in self.procs:
            g.close()
