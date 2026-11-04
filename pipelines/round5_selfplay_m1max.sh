#!/bin/zsh
# Round 5 self-play on the M1 Max.
# Model: round-3 shared model (release asset shared_round3_6x96.pt -> checkpoints/round3_merged_20261102/final.pt).
# Training moves: 1,600 visits on 12.5% of moves. Other moves: 200 visits, tree reuse, sampled by visit share
# in the first 30 moves (opening diversity). No resignation. GNU Go crashes are retried (fixed in this version).
# Resume: run the same command again; finished games are skipped.
set -e
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
GAMES=${GAMES:-12000}
R=selfplay_round5_1600v_p125_f200_seed20507105
M=checkpoints/round3_merged_20261102/final.pt
[ -f $M ] || { echo "missing $M — see docs/M1MAX_ROUND5.md step 2"; exit 1; }
caffeinate -i .venv/bin/python -u scripts/selfplay_parallel.py --run $R --games $GAMES --seed 20507105 \
  --model $M --reuse-tree --visits 1600 --fast-visits 200 --full-prob 0.125 --fast-opening-sample \
  > runs/$R.log 2>&1
echo "done: $(find runs/experiments/$R -name 'game_*.json' | wc -l) games"
