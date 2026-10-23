import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import urllib.request

import numpy as np
import pytest
import torch
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import play_server
from alphago import Board
from alphago.mcts import MCTS, TreeBudget


class FlatNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.cfg = SimpleNamespace(ownership=False)

    def forward(self, x):
        return torch.zeros((len(x), 362), device=x.device), torch.zeros(len(x), device=x.device)


def flat_evaluator(boards):
    return np.zeros((len(boards), 362), np.float32), np.zeros(len(boards), np.float32)


@pytest.fixture
def game():
    g = play_server.Game(FlatNet(), playouts=16)
    g.model_name = 'test'
    g._evaluator = flat_evaluator
    yield g
    g.close()


def wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('live analysis did not reach expected state')


def test_root_continues_without_reset_and_cancel_restores_visit_counts():
    board = Board(19)
    search = MCTS(flat_evaluator, playouts=32, batch=8)
    root = search.search(board)
    assert search.search(board, root=root) is root
    assert root.N.sum() == 64
    stop = threading.Event()
    stop.set()
    search.search(board, root=root, stop_event=stop)
    assert root.N.sum() == 64 and np.all(root.N >= 0)


def test_manual_live_accumulates_then_invalidates_on_move(game):
    with game.lock:
        game.set_mode('both')
    wait_for(lambda: game.analysis is not None and game.analysis['playouts'] >= 32)
    with game.lock:
        first = game.analysis['playouts']
    wait_for(lambda: game.analysis['playouts'] > first)
    with game.lock:
        game.play(0)
        revision = game.revision
        assert game.analysis is None
    wait_for(lambda: game.analysis is not None and game.analysis['revision'] == revision)
    with game.lock:
        assert game.analysis['moves'] == 1
        assert all(c['action'] != 0 for c in game.analysis['candidates'])
        game.set_live(False)
    time.sleep(.05)
    assert game.analysis is None


def test_move_cancels_inflight_old_position_without_waiting_for_network(game):
    entered, release = threading.Event(), threading.Event()
    def blocked(boards):
        if boards[0].move_number == 0:
            entered.set()
            assert release.wait(5)
        return flat_evaluator(boards)
    game._evaluator = blocked
    with game.lock:
        game.set_mode('both')
    assert entered.wait(5)
    try:
        with game.lock:
            game.play(0)
            revision = game.revision
            assert game.analysis is None
    finally:
        release.set()
    wait_for(lambda: game.analysis is not None)
    assert game.analysis['revision'] == revision and game.analysis['moves'] == 1


def test_tree_limit_pauses_without_deleting_statistics_then_move_restarts(game, monkeypatch):
    monkeypatch.setattr(play_server, 'LIVE_NODE_LIMIT', 37)
    with game.lock:
        game.set_mode('both')
    wait_for(lambda: game.analysis is not None and game.analysis['limit_reached'])
    with game.lock:
        snapshot = json.loads(json.dumps(game.analysis))
    assert snapshot['nodes'] == 37
    assert snapshot['limit_reason'] == 'nodes'
    time.sleep(.08)
    assert game.analysis == snapshot  # stopped, no restart or zeroed root stats
    with game.lock:
        game.play(0)
        revision = game.revision
    wait_for(lambda: game.analysis is not None and game.analysis['revision'] == revision)
    assert game.analysis['moves'] == 1


def test_node_limit_counts_retained_nodes_and_preserves_all_children():
    board = Board(19)
    budget = TreeBudget(max_nodes=37)
    search = MCTS(flat_evaluator, playouts=4096, batch=16)
    root = search.search(board, budget=budget)
    def retained(node):
        return 1 + sum(retained(c) for c in node.children.values())
    assert budget.nodes == retained(root) == 37
    assert budget.limit_reason == 'nodes'
    before_n, before_w = root.N.copy(), root.W.copy()
    children = dict(root.children)
    search.search(board, root=root, budget=budget)
    np.testing.assert_array_equal(root.N, before_n)
    np.testing.assert_array_equal(root.W, before_w)
    assert root.children == children


def test_byte_limit_stops_before_budget_is_exceeded():
    budget = TreeBudget(max_nodes=200_000, max_bytes=150_000)
    root = MCTS(flat_evaluator, playouts=4096, batch=16).search(Board(19), budget=budget)
    assert budget.limit_reason == 'memory'
    assert 1 < budget.nodes < 200_000
    assert budget.bytes <= budget.max_bytes
    assert root.children  # no pruning to satisfy the byte limit


def test_manual_http_plays_one_ply_and_undoes_one_ply(game):
    server = ThreadingHTTPServer(('127.0.0.1', 0), play_server.make_handler(game))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def post(path, payload):
        request = urllib.request.Request(f'http://127.0.0.1:{server.server_port}' + path,
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)
    try:
        state = post('/api/mode', {'human': 'both'})
        assert state['human'] == 'both' and state['live_enabled']
        post('/api/live', {'enabled': False})
        assert post('/api/play', {'action': 0})['moves'] == 1
        assert post('/api/play', {'action': 1})['moves'] == 2
        assert post('/api/undo', {})['moves'] == 1
        assert post('/api/genmove', {})['moves'] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
