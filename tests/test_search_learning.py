import copy

import numpy as np
import pytest
import torch

from alphago import Board
from alphago.data import symmetry_tables
from alphago.mcts import Node
from alphago.models import GoNet, NetConfig
from alphago.search_learning import (SearchData, SearchPlayConfig, choose_action, generate_game,
                                     losses, save_game, search_turn, visit_target)


def pass_evaluator(boards):
    actions = boards[0].size**2 + 1
    logits = np.full((len(boards), actions), -100., np.float32)
    logits[:, -1] = 100.
    return logits, np.zeros(len(boards), np.float32)


def judge(moves):
    assert moves[-2:] == [25, 25]
    return -7.5, [], np.full(25, -1, np.int8)


def pass_game():
    cfg = SearchPlayConfig(visits=400, batch=32, board_size=5, noise_fraction=0.)
    return generate_game(pass_evaluator, cfg, np.random.default_rng(0), judge)


def test_policy_target_is_visit_distribution_including_pass_not_chosen_move():
    root = Node(np.array([1, 25]), np.array([.5, .5]), 0.)
    root.N[:] = [100, 300]
    pi = visit_target(root, 26)
    assert pi[1] == .25 and pi[25] == .75
    assert pi.sum() == 1
    assert choose_action(pi, np.random.default_rng(0), 0.) == 25
    root.N[0] = -1
    with pytest.raises(ValueError): visit_target(root, 26)


def test_actual_400_searches_and_terminal_outcome_have_correct_perspectives(tmp_path):
    rec = pass_game()
    assert rec['moves'].tolist() == [25, 25]
    assert rec['visits'].tolist() == [400, 400]
    assert rec['judged'] and not rec['capped']
    assert rec['features'].shape == (2, 34, 5, 5)
    path = tmp_path/'game.npz'
    save_game(path, rec)
    data = SearchData([path], board_size=5)
    _, pi, z, own = data.batch(data.rows)
    np.testing.assert_array_equal(z, [-1, 1])
    np.testing.assert_array_equal(pi[:, 25], [1, 1])
    assert np.all(own[0] == -1) and np.all(own[1] == 1)


def test_capped_and_failed_games_never_become_training_labels(tmp_path):
    cfg = SearchPlayConfig(visits=400, batch=32, board_size=5, noise_fraction=0., max_moves=1)
    def forbidden(moves): raise AssertionError('Do not adjudicate an unfinished game')
    capped = generate_game(pass_evaluator, cfg, np.random.default_rng(0), forbidden)
    assert capped['capped'] and not capped['judged']
    path = tmp_path/'capped.npz'; save_game(path, capped)
    with pytest.raises(ValueError, match='No complete'): SearchData([path], board_size=5)
    cfg.max_moves = 10
    failed = generate_game(pass_evaluator, cfg, np.random.default_rng(0), forbidden)
    assert not failed['capped'] and not failed['judged'] and 'judge_error' in failed


def test_augmentation_rotates_soft_policy_ownership_and_features_together(tmp_path):
    rec = pass_game()
    rec['features'][:] = 0
    rec['features'][:, 0, 0, 1] = 1
    rec['policy'][:] = 0
    rec['policy'][:, 1] = .8
    rec['policy'][:, 25] = .2
    rec['ownership'][:] = 0; rec['ownership'][1] = 1
    path = tmp_path/'game.npz'; save_game(path, rec)
    data = SearchData([path], board_size=5)
    class FixedSymmetry:
        def __init__(self, s): self.s = s
        def integers(self, low, high, size): return np.full(size, self.s)
    for s in range(8):
        f, pi, _, own = data.batch(data.rows[:1], FixedSymmetry(s), augment=True)
        action = symmetry_tables(5)[s, 1]
        assert np.isclose(pi[0, action], .8) and np.isclose(pi[0, 25], .2)
        assert f[0, 0].ravel()[action] == 1
        assert own[0].ravel()[action] == 1


def test_joint_loss_updates_both_trunks_and_keeps_anchor_frozen():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    policy = GoNet(NetConfig(board_size=5, blocks=1, channels=8))
    value = GoNet(NetConfig(board_size=5, blocks=1, channels=8, policy=False, ownership=True))
    anchor = copy.deepcopy(policy).eval()
    for p in anchor.parameters(): p.requires_grad_(False)
    original_p = policy.stem[0].weight.detach().clone()
    original_v = value.stem[0].weight.detach().clone()
    original_anchor = copy.deepcopy(anchor.state_dict())
    rng = np.random.default_rng(5)
    batch = (rng.integers(0, 2, (4, 34, 5, 5), dtype=np.uint8),
             np.full((4, 26), 1/26, np.float32), np.array([1,-1,1,-1], np.float32),
             rng.choice([-1., 1.], size=(4,5,5)).astype(np.float32))
    opt = torch.optim.SGD(list(policy.parameters()) + list(value.parameters()), lr=.01)
    total, metrics = losses(policy, value, anchor, batch, torch.device('cpu'))
    assert set(metrics) == {'policy_ce', 'value_mse', 'anchor_kl', 'ownership_mse'}
    total.backward(); opt.step()
    assert not torch.equal(original_p, policy.stem[0].weight)
    assert not torch.equal(original_v, value.stem[0].weight)
    for k, v in original_anchor.items(): assert torch.equal(v, anchor.state_dict()[k])


def test_search_noise_changes_priors_but_keeps_fixed_budget_and_legal_target():
    def uniform(boards):
        return np.zeros((len(boards), 26), np.float32), np.zeros(len(boards), np.float32)
    b = Board(5, superko=True)
    a, pi, _, visits = search_turn(b, uniform, SearchPlayConfig(board_size=5), np.random.default_rng(2))
    assert visits == 400 and b.move_number == 0
    assert pi.shape == (26,) and np.isclose(pi.sum(), 1)
    assert b.legal_mask()[a]
