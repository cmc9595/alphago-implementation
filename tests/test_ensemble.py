import numpy as np
import torch
from alphago.data import transform
from alphago.mcts import average_value, transform_tensor, NetEvaluator, MCTS


class ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
    def forward(self,x):
        n=x.shape[-1]
        return torch.ones((len(x),n*n+1))*self.weight, torch.tanh(self.weight*(x[:,0,0,0]+2*x[:,0,-1,0]))


def test_tensor_and_numpy_symmetries_agree():
    x=np.arange(2*3*5*5).reshape(2,3,5,5).copy()
    for s in range(8):
        np.testing.assert_array_equal(transform_tensor(torch.from_numpy(x),s).numpy(),transform(x,s))


def test_d4_value_is_invariant_without_changing_player_view():
    rng=np.random.default_rng(5)
    x=torch.from_numpy(rng.normal(size=(7,2,5,5)).astype(np.float32))
    net=ToyNet()
    expected=average_value(net,x,8)
    for s in range(8):
        torch.testing.assert_close(average_value(net,transform_tensor(x,s),8),expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(average_value(net,x,1),net(x)[1])


def test_value_averaging_leaves_policy_logits_unchanged(monkeypatch):
    x=np.random.default_rng(0).integers(0,2,(2,34,5,5),dtype=np.uint8)
    monkeypatch.setattr('alphago.mcts.features_batch',lambda boards:x[:len(boards)])
    net=ToyNet()
    p1,v1=NetEvaluator(net,value_symmetries=1)([None,None])
    p8,v8=NetEvaluator(net,value_symmetries=8)([None,None])
    np.testing.assert_array_equal(p1,p8)
    np.testing.assert_allclose(v8,average_value(net,torch.from_numpy(x).float(),8).numpy())


def test_search_time_budget_keeps_actual_visit_counts(monkeypatch):
    from alphago import Board
    clock=iter(np.arange(10000)*.01)
    monkeypatch.setattr('alphago.mcts.time.perf_counter',lambda:next(clock))
    def evaluator(boards):
        return np.zeros((len(boards),82),np.float32),np.zeros(len(boards),np.float32)
    root=MCTS(evaluator,playouts=500,batch=8,time_limit=.025).search(Board(9))
    assert 0<root.N.sum()<500
    assert root.N.sum()%8==0
