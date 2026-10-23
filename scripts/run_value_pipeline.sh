#!/bin/zsh
# Waits for value-data generation and the human-ownership job, then trains both value-net
# variants back to back (KataGo-style with ownership head, AlphaGo-style without).
cd "$(dirname "$0")/.."
until grep -q '^done' data/value/gen/run.log 2>/dev/null && ! pgrep -f prepare_human_ownership.py >/dev/null; do
  sleep 60
done
echo "$(date) data ready, training value_own"
.venv/bin/python scripts/train_value.py --run value_own --ownership-weight 1.5
echo "$(date) training value_plain"
.venv/bin/python scripts/train_value.py --run value_plain --ownership-weight 0
echo "$(date) all done"
