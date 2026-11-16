import pytest
import torch

from alphago.models import GlobalPoolBias, GoNet, NetConfig, expand_blocks
from alphago.shared_training import training_mode


def test_expansion_preserves_all_heads_and_old_parameters():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    old = GoNet(NetConfig(board_size=5, blocks=2, channels=8, ownership=True, score=True)).eval()
    new = expand_blocks(old, 4)
    assert old.cfg.blocks == 2 and new.cfg.blocks == 4
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    x = torch.rand(3, 34, 5, 5)
    with torch.no_grad():
        for a, b in zip(old.forward_heads(x), new.forward_heads(x)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    training_mode(new)
    sum(t.square().mean() for t in new.forward_heads(x)).backward()
    # Added blocks can learn despite starting as identity functions.
    assert new.trunk[2].bn2.weight.grad.abs().sum() > 0


def test_expansion_rejects_non_increasing_depth():
    old = GoNet(NetConfig(blocks=1, channels=4))
    with pytest.raises(ValueError):
        expand_blocks(old, 1)


@pytest.mark.parametrize('depth', [6, 10])
def test_global_pool_expansion_preserves_outputs_and_can_learn(depth):
    torch.manual_seed(3)
    old = GoNet(NetConfig(board_size=5, blocks=6, channels=8, ownership=True)).eval()
    new = expand_blocks(old, depth, global_pool=True)
    assert new.cfg.global_pool
    assert sum(b.global_context is not None for b in new.trunk) == depth // 3
    x = torch.rand(3, 34, 5, 5)
    with torch.no_grad():
        for before, after in zip(old.forward_all(x), new.forward_all(x)):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    training_mode(new)
    sum(y.square().mean() for y in new.forward_all(x)).backward()
    for block in new.trunk:
        if block.global_context is not None:
            assert block.global_context.projection.weight.grad.abs().sum() > 0


def test_global_pool_uses_whole_board_and_broadcasts_channel_bias():
    pool = GlobalPoolBias(1)
    with torch.no_grad():
        pool.projection.weight.fill_(1)
    x = torch.zeros(1, 1, 5, 5)
    a = pool(x)
    x[0, 0, 4, 4] = 1
    b = pool(x)
    assert b.shape == (1, 1, 1, 1)
    torch.testing.assert_close(b - a, torch.tensor([[[[1.04]]]]))
    torch.testing.assert_close(pool(x.flip((2, 3))), b)
