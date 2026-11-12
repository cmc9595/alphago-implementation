#!/bin/zsh
# Round 6 self-play on this Mac (M3 Pro). Model: round-5 shared model (beat round 3 120/200).
# Same settings as round 5: 1,600 visits on 12.5% of moves (training moves), 200 visits elsewhere with tree reuse,
# first 30 moves sampled by visit share, no resignation. Resume: run the same command again.
set -e
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
GAMES=${GAMES:-5000}
R=selfplay_round6_1600v_p125_f200_seed20600000
M=checkpoints/round5/final.pt
caffeinate -i .venv/bin/python -u scripts/selfplay_parallel.py --run $R --games $GAMES --seed 20600000 \
  --model $M --reuse-tree --visits 1600 --fast-visits 200 --full-prob 0.125 --fast-opening-sample \
  > runs/$R.log 2>&1 &
echo $! > runs/$R.pid   # progress_selfplay_parallel.py reads this to show 실행 중
wait $!
echo "done: $(find runs/experiments/$R -name 'game_*.json' | wc -l) games"
