from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from progress_train_search import display_width, metrics_table


def test_table_includes_baseline_recent_history_and_aligned_korean_columns():
    info = {'baseline': {'selfplay_value_mse': .7, 'selfplay_policy_ce': 2.3}}
    history = [{'step': 1000 * i, 'seconds': 60 * i, 'loss': 3.2,
                'selfplay_value_mse': .7 - .01 * i, 'selfplay_policy_ce': 2.3,
                'selfplay_top1_match': .62} for i in range(1, 4)]
    rows = metrics_table(info, history, rows=2)
    text = '\n'.join(rows)
    assert '기준' in text and '0.7000' in text
    assert '1,000' not in text and '2,000' in text and '3,000' in text
    assert '62.0%' in text and '00:03:00' in text
    assert len({display_width(line) for line in rows}) == 1
    assert '1,000' in '\n'.join(metrics_table(info, history, rows=0))


def test_empty_and_baseline_only_tables():
    assert metrics_table({}, []) == []
    text = '\n'.join(metrics_table({'baseline': {'selfplay_value_mse': .5}}, []))
    assert '기준' in text and '0.5000' in text
    assert '생성 가치 MSE' not in text
