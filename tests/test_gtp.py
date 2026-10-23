import torch

from alphago.gtp import GTPEngine, PolicyPlayer, action_to_vertex, vertex_to_action
from alphago.models import GoNet, NetConfig


def engine():
    torch.manual_seed(0)
    net = GoNet(NetConfig(blocks=1, channels=8)).eval()
    return GTPEngine(PolicyPlayer(net))


def test_vertex_roundtrip():
    for a in (0, 18, 19 * 19 - 1, 3 * 19 + 15, 361):
        assert vertex_to_action(action_to_vertex(a, 19), 19) == a
    assert action_to_vertex(0, 19) == "A1" and action_to_vertex(8, 19) == "J1"  # no 'I' column


def test_session():
    e = engine()
    assert e.handle("protocol_version") == (True, "2")
    assert e.handle("play B D4")[0]
    assert not e.handle("play W D4")[0]  # occupied
    ok, mv = e.handle("genmove W")
    assert ok and mv != "D4"
    assert e.handle("undo")[0] and e.handle("undo")[0]
    assert not e.handle("undo")[0]
    assert e.handle("boardsize 9") == (False, "unacceptable size")
    assert e.handle("final_score") == (True, "W+7.5")
