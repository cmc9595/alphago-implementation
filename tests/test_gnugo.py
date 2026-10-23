import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from alphago.gnugo import Referee, action, ownership, parse_score, vertex
from alphago.models import GoNet, NetConfig
from alphago.selfplay import SelfPlayConfig, play_batch

needs_gnugo = pytest.mark.skipif(shutil.which("gnugo") is None, reason="GNU Go not installed")
USER_GAME = [action(v) for v in json.loads((Path(__file__).parent / "data_user_game.json").read_text())]


def test_parse_and_vertices():
    assert parse_score("B+7.5") == 7.5 and parse_score("W+3.5") == -3.5 and parse_score("0") == 0
    for a in (0, 8, 360, 361):
        assert action(vertex(a)) == a


@needs_gnugo
def test_referee_removes_dead_stones():
    # A real game whose raw Tromp-Taylor count said W+39.5 because six dead white stones sat in
    # Black's area; GNU Go removes them and Black wins.
    ref = Referee(procs=1, level=5)
    try:
        (score, dead), = ref.judge_many([USER_GAME])
    finally:
        ref.close()
    assert score == 7.5
    assert {vertex(a) for a in dead} == {"D7", "G7", "F5", "G5", "K4", "J3", "J2"}
    own = ownership(USER_GAME, dead)
    assert own[action("J3")] == 1  # dead white stone -> Black's point
    assert own[action("D7")] == -1  # dead black stone -> White's point


@needs_gnugo
def test_play_batch_with_referee():
    torch.manual_seed(0)
    net = GoNet(NetConfig(blocks=1, channels=8)).eval()
    ref = Referee(procs=2, level=1)
    try:
        recs = play_batch(net, net, 4, np.random.default_rng(0), torch.device("cpu"),
                          SelfPlayConfig(max_moves=60), referee=ref, with_ownership=True)
    finally:
        ref.close()
    for r in recs:
        assert r.judged == "gnugo" and r.cleanup == []
        assert r.ownership is not None and r.ownership.shape == (361,)
