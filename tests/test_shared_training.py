import sys
from pathlib import Path
import numpy as np
import torch

from alphago import Board
from alphago.models import GoNet,NetConfig
from alphago.mcts import NetEvaluator
from alphago.shared_training import supervised_losses,training_mode


def test_task_masks_do_not_train_generated_policy_or_human_value_ownership():
    policy=torch.zeros(4,26,requires_grad=True)
    value=torch.zeros(4,requires_grad=True)
    own=torch.zeros(4,5,5,requires_grad=True)
    total,_=supervised_losses((policy,value,own),torch.tensor([1,2]),torch.tensor([1.,-1.]),torch.ones(2,5,5))
    total.backward()
    assert policy.grad[:2].abs().sum()>0 and policy.grad[2:].abs().sum()==0
    assert value.grad[:2].abs().sum()==0 and value.grad[2:].abs().sum()>0
    assert own.grad[:2].abs().sum()==0 and own.grad[2:].abs().sum()>0


def test_head_warmup_preserves_policy_and_trunk_then_joint_training_updates_trunk():
    torch.set_num_threads(2);torch.manual_seed(0)
    net=GoNet(NetConfig(board_size=5,blocks=1,channels=8,ownership=True))
    training_mode(net,heads_only=True)
    x=torch.rand(4,34,5,5)
    initial=net.stem[0].weight.detach().clone()
    mean=net.stem[1].running_mean.clone()
    opt=torch.optim.SGD(net.parameters(),lr=.01)
    total,_=supervised_losses(net.forward_all(x),torch.tensor([1,2]),torch.tensor([1.,-1.]),torch.ones(2,5,5))
    total.backward();opt.step()
    assert torch.equal(initial,net.stem[0].weight)
    assert torch.equal(mean,net.stem[1].running_mean)
    assert net.policy_fc.weight.grad is None
    training_mode(net,heads_only=False);opt.zero_grad(set_to_none=True)
    total,_=supervised_losses(net.forward_all(x),torch.tensor([1,2]),torch.tensor([1.,-1.]),torch.ones(2,5,5))
    total.backward();opt.step()
    assert not torch.equal(initial,net.stem[0].weight)
    assert torch.equal(mean,net.stem[1].running_mean)
    assert net.policy_fc.weight.grad is not None


class CountingNet(GoNet):
    def __init__(self,cfg):super().__init__(cfg);self.calls=0
    def forward(self,*args,**kwargs):self.calls+=1;return super().forward(*args,**kwargs)


def test_shared_search_evaluator_uses_one_trunk_forward():
    cfg=NetConfig(board_size=5,blocks=1,channels=8)
    shared=CountingNet(cfg)
    logits,values=NetEvaluator(shared)([Board(5)])
    assert logits.shape==(1,26) and values.shape==(1,) and shared.calls==1
    policy,value=CountingNet(cfg),CountingNet(cfg)
    NetEvaluator(policy,value)([Board(5)])
    assert policy.calls==value.calls==1


def test_match_shared_spec_keeps_same_net_identity_and_requested_batch(monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    import match
    net=CountingNet(NetConfig(board_size=5,blocks=1,channels=8))
    monkeypatch.setattr(match,'net',lambda path:net)
    engine=match.Engine('mcts:unused.pt:400:batch=32',7.5)
    assert engine.player.mcts.batch==32
    assert engine.player.evaluator.policy_net is engine.player.evaluator.value_net
