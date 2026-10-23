"""Local browser board for playing against (and inspecting) a policy network.

    python scripts/play_server.py --model checkpoints/sl_6x96/final.pt
    open http://localhost:8765

Standard library HTTP server only; the page shows the policy heatmap and value estimate.
"""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import torch

from alphago import Board, Color
from alphago.gtp import PolicyPlayer, action_to_vertex, load_net, vertex_to_action
from alphago.gnugo import Referee
from alphago.mcts import MCTS, MCTSPlayer, TreeBudget

MAX_PLAYOUTS = 12800
SEARCH_BATCH = 32
ANALYSIS_CANDIDATES = 12
LIVE_NODE_LIMIT = 200_000
LIVE_TREE_BYTES = 2 * 1024**3

UI = Path(__file__).resolve().parent.parent / "ui" / "index.html"


class Game:
    def __init__(self, net, value_net=None, playouts=1600, temperature=0.0):
        self.net, self.value_net = net, value_net
        self.policy = PolicyPlayer(net, temperature=temperature)
        self.lock = threading.RLock()
        self.compute_lock = threading.Lock()
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._cancel = threading.Event()
        self.revision = 0
        self._static_state = None
        self.live_enabled = False
        self.live_error = None
        self.live_last_seen = 0.
        self.analysis = None
        self.set_playouts(playouts)
        self.new(human="B", komi=7.5)
        self._worker = threading.Thread(target=self._live_worker, daemon=True, name="live-mcts")
        self._worker.start()

    def set_playouts(self, n: int):
        """Engine strength: 0 = raw policy, otherwise MCTS with n playouts per move."""
        self._static_state = None
        self.playouts = n
        self.player = MCTSPlayer(self.net, self.value_net, playouts=n, batch=SEARCH_BATCH) if n > 0 else self.policy
        self.engine_name = f"MCTS {n}회 탐색" if n > 0 else "정책망 단독 (탐색 없음)"

    def invalidate(self):
        self._cancel.set()
        self._cancel = threading.Event()
        self.revision += 1
        self.analysis = None
        self._static_state = None
        self._judge_key = None
        self._wake.set()

    def set_mode(self, human):
        if human not in ("B", "W", "both"):
            raise ValueError("invalid playing mode")
        self.human = human
        self.invalidate()
        if human == "both":
            self.set_live(True)

    def set_live(self, enabled):
        self.live_enabled = bool(enabled)
        self.live_error = None
        self.invalidate()
        self.touch()

    def touch(self):
        self.live_last_seen = time.monotonic()
        self._wake.set()

    def analysis_snapshot(self, root, *, continuous=False, seconds=0., budget=None):
        total = int(root.N.sum())
        cands = []
        order = [i for i in np.argsort(-root.N) if root.N[i] > 0][:ANALYSIS_CANDIDATES]
        for i in order:
            pv, node = [int(root.moves[i])], root.children.get(int(i))
            while node is not None and node.N.sum() > 0 and len(pv) < 10:
                j = int(np.argmax(node.N))
                pv.append(int(node.moves[j]))
                node = node.children.get(j)
            q = float(root.W[i] / root.N[i])
            cands.append({"action": int(root.moves[i]), "vertex": action_to_vertex(int(root.moves[i]), 19),
                          "visits": int(root.N[i]), "visit_share": round(float(root.N[i] / max(total, 1)), 4),
                          "win": round((q + 1) / 2, 3), "prior": round(float(root.P[i]), 3),
                          "pv": [action_to_vertex(a, 19) for a in pv]})
        value = float(root.W.sum() / total) if total else root.value
        self.analysis = {"moves": len(self.history), "revision": self.revision, "playouts": total,
                         "continuous": continuous, "seconds": round(seconds, 2),
                         "nodes": budget.nodes if budget else None,
                         "tree_bytes": budget.bytes if budget else None,
                         "node_limit": budget.max_nodes if budget else None,
                         "byte_limit": budget.max_bytes if budget else None,
                         "limit_reached": bool(budget and budget.limit_reason),
                         "limit_reason": budget.limit_reason if budget else None,
                         "black_win": round((1 + value * (1 if self.board.to_move == Color.BLACK else -1)) / 2, 3),
                         "candidates": cands}

    def analyze(self, playouts: int):
        """One-shot analysis; cancel the continuous worker before sharing inference."""
        self.invalidate()
        with self.compute_lock:
            root = MCTS(self.policy_evaluator(), playouts=playouts, batch=SEARCH_BATCH).search(self.board)
        self.analysis_snapshot(root)

    def _live_worker(self):
        root, root_revision, seconds, budget = None, -1, 0., None
        while not self._shutdown.is_set():
            with self.lock:
                active = (self.live_enabled and not self.board.game_over and
                          (self.human == "both" or (self.board.to_move == Color.BLACK) == (self.human == "B")) and
                          time.monotonic() - self.live_last_seen < 10)
                if active:
                    revision, board, cancel = self.revision, self.board.copy(), self._cancel
                keep_tree = self.live_enabled and not self.board.game_over and self.revision == root_revision
            if not active:
                # Disconnection pauses computation but preserves the bounded tree.
                # Moves, explicit stop, and game end release it.
                if not keep_tree:
                    root, root_revision, budget = None, -1, None
                self._wake.wait(.5); self._wake.clear()
                continue
            if revision != root_revision:
                root, root_revision, seconds = None, revision, 0.
                budget = TreeBudget(LIVE_NODE_LIMIT, LIVE_TREE_BYTES)
            if budget.limit_reason:
                self._wake.wait(.5); self._wake.clear()
                continue
            try:
                with self.compute_lock:
                    started = time.perf_counter()
                    if cancel.is_set():
                        continue
                    root = MCTS(self.policy_evaluator(), playouts=256, batch=SEARCH_BATCH, time_limit=.18).search(
                        board, root=root, stop_event=cancel, budget=budget)
                    seconds += time.perf_counter() - started
                budget.capacity(board.size, 1)
                with self.lock:
                    if revision != self.revision or cancel.is_set():
                        root = None
                        continue
                    if root.N.sum() or budget.limit_reason:
                        self.analysis_snapshot(root, continuous=True, seconds=seconds, budget=budget)
                self._shutdown.wait(.01)  # give interactive requests a chance between chunks
            except Exception as exc:
                with self.lock:
                    if revision == self.revision:
                        self.live_error = str(exc)
                        self.live_enabled = False
                root, root_revision = None, -1

    def close(self):
        self._cancel.set()
        self._shutdown.set()
        self._wake.set()
        self._worker.join(timeout=5)

    def policy_evaluator(self):
        from alphago.mcts import NetEvaluator

        if not hasattr(self, "_evaluator"):
            self._evaluator = NetEvaluator(self.net, self.value_net)
        return self._evaluator

    def judge(self):
        """GNU Go's dead-stone verdict for a finished game (cached per position)."""
        key = (len(self.history), self.komi)
        if getattr(self, "_judge_key", None) != key:
            self._judge_key, self._judge = key, None
            try:
                ref = Referee(procs=1, level=10, komi=self.komi)
                try:
                    (verdict,) = ref.judge_many([self.history])
                finally:
                    ref.close()
                if verdict:
                    score, dead = verdict
                    self._judge = {"score": score, "dead": dead}
            except Exception:
                pass  # GNU Go missing: the page falls back to the raw count
        return self._judge

    def ownership(self):
        """Territory estimate from the value net's ownership head (Black's view), or None."""
        net = self.value_net if self.value_net is not None else self.net
        if not getattr(net.cfg, "ownership", False):
            return None
        b = self.board
        with torch.no_grad():
            x = torch.from_numpy(b.features()[None]).to(next(net.parameters()).device).float()
            own = net.ownership(x)[0].float().cpu().numpy().ravel()
        if b.to_move != Color.BLACK:
            own = -own  # the head answers for the side to move
        stones = b.stones().ravel()
        # Stones predicted to end up in the opponent's area are dead.
        dead = [int(a) for a in np.flatnonzero(((stones == 1) & (own < -0.5)) | ((stones == 2) & (own > 0.5)))]
        black = int((own > 0.33).sum())
        white = int((own < -0.33).sum())
        return {"map": [round(float(o), 2) for o in own], "dead": dead,
                "score": round(black - white - self.komi, 1)}

    def score_lead(self):
        """Score head estimate: final margin for Black (komi included), or None if the net has no score head."""
        net = self.value_net if self.value_net is not None else self.net
        if not getattr(net.cfg, "score", False):
            return None
        from alphago.models import SCORE_SCALE
        b = self.board
        with torch.no_grad():
            x = torch.from_numpy(b.features()[None]).to(next(net.parameters()).device).float()
            s = float(net.forward_heads(x)[3][0]) * SCORE_SCALE
        return round(s if b.to_move == Color.BLACK else -s, 1)

    def new(self, human, komi):
        if human not in ("B", "W", "both"):
            raise ValueError("invalid playing mode")
        self.invalidate()
        self.human = human
        self.komi = komi
        self.history: list[int] = []
        self.board = Board(19, komi, superko=True)

    def rebuild(self):
        self.invalidate()
        self.board = Board(19, self.komi, superko=True)
        for a in self.history:
            self.board.play(a)

    def play(self, a):
        if self.board.game_over:
            raise ValueError("game is over")
        if not self.board.is_legal(a):
            raise ValueError("illegal move")
        self.invalidate()
        self.board.play(a)
        self.history.append(a)

    def engine_move(self):
        if not self.board.game_over:
            with self.compute_lock:
                move = self.player.genmove(self.board)
            self.play(move)

    def state(self):
        if self._static_state is None:
            with self.compute_lock:
                self._static_state = self._position_state()
        state = dict(self._static_state)
        state.update(evaluation_batch=SEARCH_BATCH, human=self.human, revision=self.revision, engine=self.engine_name, playouts=self.playouts,
                     live_enabled=self.live_enabled, live_error=self.live_error,
                     live_running=self.live_enabled and not self.board.game_over and
                     (self.human == "both" or (self.board.to_move == Color.BLACK) == (self.human == "B")) and
                     time.monotonic() - self.live_last_seen < 10 and
                     not bool(self.analysis and self.analysis.get("limit_reached")),
                     analysis=self.analysis if self.analysis and self.analysis.get("revision") == self.revision else None)
        return state

    def _position_state(self):
        b = self.board
        probs, value = self.player.evaluate(b)
        if self.value_net is not None:  # the bar should show the dedicated value net, not the policy's head
            with torch.no_grad():
                x = torch.from_numpy(b.features()[None]).to(next(self.value_net.parameters()).device).float()
                value = float(self.value_net(x)[1][0])
        legal = b.legal_mask()
        black_to_move = b.to_move == Color.BLACK
        # Value head is from the side to move's perspective; report Black's win probability.
        black_win = (value + 1) / 2 if black_to_move else (1 - value) / 2
        top = np.argsort(-np.where(legal.astype(bool), probs, 0))[:5]
        return {
            "stones": b.stones().ravel().tolist(),
            "to_move": "B" if black_to_move else "W",
            "human": self.human,
            "last": b.last_action,
            "moves": len(self.history),
            "history": [action_to_vertex(a, 19) for a in self.history],
            "captures": {"B": b.captures(Color.BLACK), "W": b.captures(Color.WHITE)},
            "policy": [round(float(p), 4) for p in probs],
            "legal": legal.tolist(),
            "black_win": round(float(black_win), 3),
            "top": [{"action": int(a), "vertex": action_to_vertex(int(a), 19), "p": round(float(probs[a]), 3)} for a in top],
            "game_over": b.game_over,
            "score": round(float(b.score()), 1) if b.game_over else None,
            "judge": self.judge() if b.game_over else None,
            "territory": self.ownership(),
            "score_lead": self.score_lead(),
            "komi": self.komi,
            "engine": self.engine_name,
            "model": self.model_name,
            "playouts": self.playouts,
            "search": self.search_info(),
            "analysis": None,
        }

    def search_info(self):
        root = getattr(self.player, "last_root", None)
        if root is None:
            return None
        return [dict(s, vertex=action_to_vertex(s["action"], 19)) for s in MCTS.summary(root, 6)]


def make_handler(game: Game):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send(200, UI.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                with game.lock:
                    game.touch()
                    self.send(200, game.state())
            else:
                self.send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
            try:
                with game.lock:
                    if self.path == "/api/new":
                        game.new(req.get("human", "B"), float(req.get("komi", 7.5)))
                        if game.human == "both":
                            game.set_live(True)
                        if game.human == "W":
                            game.engine_move()
                    elif self.path == "/api/play":
                        game.play(int(req["action"]))
                        if game.human != "both" and req.get("reply", True):
                            game.engine_move()
                    elif self.path == "/api/mode":
                        game.set_mode(req["human"])
                        if game.human != "both" and (game.board.to_move == Color.BLACK) != (game.human == "B"):
                            game.engine_move()
                    elif self.path == "/api/live":
                        game.set_live(req.get("enabled", False))
                    elif self.path == "/api/settings":
                        game.set_playouts(max(0, min(int(req["playouts"]), MAX_PLAYOUTS)))
                    elif self.path == "/api/analyze":
                        if not game.board.game_over:
                            game.analyze(max(16, min(int(req.get("playouts", 800)), MAX_PLAYOUTS)))
                    elif self.path == "/api/load":
                        # Replay a game record: {"history": ["Q4", "D16", ...], "human": "B"|"W"}
                        game.new(req.get("human", "B"), float(req.get("komi", game.komi)))
                        for v in req["history"]:
                            game.play(vertex_to_action(v, 19))
                        if game.human == "both":
                            game.set_live(req.get("live_enabled", True))
                    elif self.path == "/api/genmove":
                        # "AI plays for me": move for the human, then let the engine answer.
                        game.engine_move()
                        human_black = game.human == "B"
                        if game.human != "both" and not game.board.game_over and (game.board.to_move == Color.BLACK) != human_black:
                            game.engine_move()
                    elif self.path == "/api/undo":
                        # Undo back to the human's turn (engine reply + human move).
                        human_black = game.human == "B"
                        while game.history:
                            game.history.pop()
                            game.rebuild()
                            if game.human == "both" or (game.board.to_move == Color.BLACK) == human_black:
                                break
                    else:
                        return self.send(404, {"error": "not found"})
                    game.touch()
                    self.send(200, game.state())
            except (ValueError, KeyError) as e:
                self.send(400, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/sl_6x96/final.pt")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--playouts", type=int, default=1600, help="initial engine strength; 0 = raw policy (changeable in the page)")
    ap.add_argument("--value-model", default=None)
    ap.add_argument("--label", default=None, help="model name shown in the page")
    args = ap.parse_args()
    value_net = load_net(args.value_model) if args.value_model else None
    game = Game(load_net(args.model), value_net, playouts=args.playouts, temperature=args.temperature)
    game.model_name = args.label or Path(args.model).parent.name + "/" + Path(args.model).stem
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(game))
    print(f"model {args.model}\nopen http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        game.close()
        server.server_close()


if __name__ == "__main__":
    main()
