#!/bin/zsh
# Value target = 0.5*outcome + 0.5*MCTS root value, on ALL 400-visit self-play positions (value fraction 1.0).
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
log "[1/3] train from round-1 model"
caffeinate -i .venv/bin/python -u scripts/train_search.py \
  --selfplay runs/experiments/selfplay_cap_3300_20261019 runs/experiments/selfplay_round2_3000_20261019 \
  --init checkpoints/search_finetune_20261019/final.pt --run searchvalue_mix05_20261022 \
  --value-fraction 1.0 --search-mix 0.5 --batch 64 --human-batch 32 --gen-batch 16 \
  --epochs 2 --eval-every 1000 --save-evals > runs/searchvalue_mix05_20261022.log 2>&1 || { log "train failed"; exit 1; }
log "[2/3] A/B vs round-1, 400 visits, 100 games"
.venv/bin/python -u scripts/match_parallel.py checkpoints/searchvalue_mix05_20261022/final.pt checkpoints/search_finetune_20261019/final.pt \
  --visits 400 --pairs 50 --seed 20261029 --name ab_searchvalue_vs_round1_400 > /dev/null 2>&1
python3 -c "import json;s=json.load(open('runs/matches/ab_searchvalue_vs_round1_400/summary.json'))['summary'];print('  ',s['A_wins'],'/',s['games'],s['ci95'])"
log "[3/3] T5 + value comparison"
.venv/bin/python -u scripts/t5_check.py checkpoints/search_finetune_20261019/final.pt checkpoints/searchvalue_mix05_20261022/final.pt 2>&1 | grep -v Warning
.venv/bin/python -u scripts/compare_value_nets.py checkpoints/search_finetune_20261019/final.pt checkpoints/searchvalue_mix05_20261022/final.pt 2>&1 | grep -v Warning
log "all done"
