"""GTP (Go Text Protocol) front end, so Sabaki / GNU Go / KataGo can talk to our engine."""

import sys

import numpy as np
import torch

from alphago import Board, Color
from alphago.models import GoNet, NetConfig, pick_device

COLS = "ABCDEFGHJKLMNOPQRST"


def action_to_vertex(a: int, n: int) -> str:
    if a == n * n:
        return "pass"
    return f"{COLS[a % n]}{a // n + 1}"


def vertex_to_action(v: str, n: int) -> int:
    v = v.strip().upper()
    if v == "PASS":
        return n * n
    x = COLS.index(v[0])
    y = int(v[1:]) - 1
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError("vertex off board")
    return y * n + x


def load_net(path: str, device=None) -> GoNet:
    device = device or pick_device()
    ck = torch.load(path, map_location=device, weights_only=False)  # our own checkpoints
    net = GoNet(NetConfig(**ck["config"])).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


class PolicyPlayer:
    """Plays the policy network's move directly (no search): the raw SL / RL policy strength."""

    def __init__(self, net: GoNet, temperature: float = 0.0, seed: int = 0):
        self.net = net
        self.device = next(net.parameters()).device
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def evaluate(self, board: Board):
        x = torch.from_numpy(board.features()[None]).to(self.device).float()
        logits, value = self.net(x)
        probs = torch.softmax(logits[0].float(), 0).cpu().numpy()
        return probs, (value[0].item() if value is not None else 0.0)

    def genmove(self, board: Board) -> int:
        probs, _ = self.evaluate(board)
        mask = board.sensible_mask().astype(bool)  # never fill own eyes; pass always allowed
        p = np.where(mask, probs, 0.0)
        n = board.size
        if p[: n * n].sum() < 1e-6:
            return n * n
        if self.temperature > 0:
            p = p ** (1 / self.temperature)
            return int(self.rng.choice(len(p), p=p / p.sum()))
        return int(p.argmax())


class GTPEngine:
    def __init__(self, player, name="alphago-implementation", komi=7.5):
        self.player = player
        self.name = name
        self.size = 19
        self.komi = komi
        self.history: list[tuple[Color, int]] = []
        self.board = Board(self.size, self.komi, superko=True)

    def _rebuild(self):
        self.board = Board(self.size, self.komi, superko=True)
        for color, a in self.history:
            self.board.set_to_move(color)
            self.board.play(a)

    def handle(self, line: str) -> tuple[bool, str]:
        parts = line.split("#", 1)[0].split()
        if not parts:
            return True, ""
        cmd, args = parts[0].lower(), parts[1:]
        n = self.size
        if cmd == "protocol_version":
            return True, "2"
        if cmd == "name":
            return True, self.name
        if cmd == "version":
            return True, "0.1"
        if cmd == "list_commands":
            return True, "\n".join(COMMANDS)
        if cmd == "known_command":
            return True, "true" if args and args[0] in COMMANDS else "false"
        if cmd == "boardsize":
            if int(args[0]) != 19:
                return False, "unacceptable size"
            self.history.clear()
            self._rebuild()
            return True, ""
        if cmd == "clear_board":
            self.history.clear()
            self._rebuild()
            return True, ""
        if cmd == "komi":
            self.komi = float(args[0])
            self._rebuild()
            return True, ""
        if cmd == "play":
            color = Color.BLACK if args[0].lower().startswith("b") else Color.WHITE
            a = vertex_to_action(args[1], n)
            self.board.set_to_move(color)
            if not self.board.is_legal(a):
                self._rebuild()
                return False, "illegal move"
            self.board.play(a)
            self.history.append((color, a))
            return True, ""
        if cmd == "genmove":
            color = Color.BLACK if args[0].lower().startswith("b") else Color.WHITE
            self.board.set_to_move(color)
            a = self.player.genmove(self.board)
            self.board.play(a)
            self.history.append((color, a))
            return True, action_to_vertex(a, n)
        if cmd == "kgs-genmove_cleanup":
            # Opponent disputes dead stones: keep playing in contested areas until they are captured.
            from alphago.selfplay import cleanup_mask

            color = Color.BLACK if args[0].lower().startswith("b") else Color.WHITE
            self.board.set_to_move(color)
            probs, _ = self.player.evaluate(self.board)
            a = int(np.argmax(np.where(cleanup_mask(self.board), probs, -1.0)))
            self.board.play(a)
            self.history.append((color, a))
            return True, action_to_vertex(a, n)
        if cmd == "undo":
            if not self.history:
                return False, "cannot undo"
            self.history.pop()
            self._rebuild()
            return True, ""
        if cmd == "showboard":
            return True, "\n" + str(self.board)
        if cmd == "final_score":
            s = self.board.score()
            return True, "0" if s == 0 else (f"B+{s:g}" if s > 0 else f"W+{-s:g}")
        if cmd == "quit":
            return True, ""
        return False, "unknown command"

    def run(self, inp=sys.stdin, out=sys.stdout):
        for line in inp:
            ok, resp = self.handle(line)
            out.write(("= " if ok else "? ") + resp + "\n\n")
            out.flush()
            if line.strip().lower() == "quit":
                break


COMMANDS = ["protocol_version", "name", "version", "list_commands", "known_command", "boardsize",
            "clear_board", "komi", "play", "genmove", "kgs-genmove_cleanup", "undo", "showboard",
            "final_score", "quit"]
