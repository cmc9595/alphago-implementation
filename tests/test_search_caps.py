import numpy as np
from dataclasses import replace
from alphago.search_learning import SearchPlayConfig, generate_game, save_game, SearchData


def test_mixed_caps_have_exact_budget_and_only_full_rows_enter_training(tmp_path):
    import alphago.search_learning as sl
    original = sl.search_turn
    calls = []
    def pass_eval(boards):
        logits = np.full((len(boards),26), -100., np.float32)
        logits[:,-1] = 100.
        return logits, np.zeros(len(boards), np.float32)
    def observed(board, evaluator, config, rng, noisy=True):
        calls.append((config.visits, noisy))
        return original(board, evaluator, config, rng, noisy)
    sl.search_turn = observed
    try:
        cfg=SearchPlayConfig(board_size=5, visits=400, fast_visits=100,
                             full_search_probability=.25, noise_fraction=0.)
        def judge(moves): return -7.5, [], np.full(25,-1,np.int8)
        found=False
        for seed in range(20):
            calls.clear()
            rec=generate_game(pass_eval,cfg,np.random.default_rng(seed),judge)
            if rec['train_mask'].sum() == 1:
                found=True
                assert sorted(calls) == [(100,False),(400,True)]
                assert sorted(rec['visits']) == [100,400]
                path=tmp_path/'mixed.npz';save_game(path,rec)
                data=SearchData([path],board_size=5)
                assert len(data.rows)==1
                ply=int(data.rows[0,1])
                assert rec['visits'][ply]==400
                _,_,z,_=data.batch(data.rows)
                assert z[0] == (-1 if ply%2==0 else 1)
                break
        assert found
    finally: sl.search_turn=original


def test_legacy_records_without_mask_remain_compatible(tmp_path):
    from test_search_learning import pass_game
    rec=pass_game(); del rec['train_mask']
    path=tmp_path/'old.npz'; save_game(path,rec)
    assert len(SearchData([path],board_size=5).rows)==2
