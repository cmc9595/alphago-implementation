#!/bin/zsh
# Round 4 self-play (run on the M1 Max). Model: round-3 shared model (release asset shared_round3_6x96.pt
# placed at checkpoints/round3_merged_20261102/final.pt). 800 visits on 6.25% of moves, 100 otherwise,
# tree reuse, no resign, and NEW: fast moves also sample by visit share in the first 30 moves (diversity).
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
R=selfplay_round4_800v_p0625_20000_seed20407102
caffeinate -i .venv/bin/python -u scripts/selfplay_parallel.py --run $R --games 20000 --seed 20407102 \
  --model checkpoints/round3_merged_20261102/final.pt --reuse-tree --visits 800 --full-prob 0.0625 \
  --fast-opening-sample > runs/$R.log 2>&1
