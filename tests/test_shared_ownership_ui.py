import sys
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from play_server import Game

class OwnNet(torch.nn.Module):
    def __init__(self, own):
        super().__init__();self.weight=torch.nn.Parameter(torch.zeros(()))
        self.cfg=SimpleNamespace(ownership=True);self.own=own
    def forward(self,x): return torch.zeros((len(x),362)),torch.zeros(len(x))
    def ownership(self,x): return torch.full((len(x),19,19),self.own)

def test_shared_ownership_is_visible_and_tracks_player_perspective():
    g=Game(OwnNet(.75),playouts=16)
    try:
        assert np.allclose(g.ownership()['map'],.75)
        g.play(0)
        assert np.allclose(g.ownership()['map'],-.75)
    finally: g.close()

def test_separate_value_ownership_still_takes_precedence():
    g=Game(OwnNet(.75),OwnNet(-.5),playouts=16)
    try: assert np.allclose(g.ownership()['map'],-.5)
    finally: g.close()
