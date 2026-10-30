#!/bin/zsh
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
log "[1/3] train: round-1 init, policy+value(0.5 outcome+0.5 MCTS)+ownership+SCORE"
caffeinate -i .venv/bin/python -u scripts/train_search.py \
  --selfplay runs/experiments/selfplay_cap_3300_20261019 runs/experiments/selfplay_round2_3000_20261019 \
  --init checkpoints/search_finetune_20261019/final.pt --run scorehead_20261023 \
  --value-fraction 1.0 --search-mix 0.5 --score-weight 0.5 --batch 64 --human-batch 32 --gen-batch 32 --gen-sources gen+gen2 \
  --epochs 2 --eval-every 1000 --save-evals > runs/scorehead_20261023.log 2>&1 || { log "train failed"; exit 1; }
log "[2/3] A/B: score-head model vs mix0.5 model (same recipe minus score head), 400 visits, 100 games"
.venv/bin/python -u scripts/match_parallel.py checkpoints/scorehead_20261023/final.pt checkpoints/searchvalue_mix05_20261022/final.pt \
  --visits 400 --pairs 50 --seed 20261030 --name ab_scorehead_vs_mix05_400 > /dev/null 2>&1
python3 -c "import json;s=json.load(open('runs/matches/ab_scorehead_vs_mix05_400/summary.json'))['summary'];print('  vs mix05',s['A_wins'],'/',s['games'],s['ci95'])"
log "[3/3] A/B vs round-1 + T5 + value table"
.venv/bin/python -u scripts/match_parallel.py checkpoints/scorehead_20261023/final.pt checkpoints/search_finetune_20261019/final.pt \
  --visits 400 --pairs 50 --seed 20261031 --name ab_scorehead_vs_round1_400 > /dev/null 2>&1
python3 -c "import json;s=json.load(open('runs/matches/ab_scorehead_vs_round1_400/summary.json'))['summary'];print('  vs round1',s['A_wins'],'/',s['games'],s['ci95'])"
.venv/bin/python -u scripts/t5_check.py checkpoints/searchvalue_mix05_20261022/final.pt checkpoints/scorehead_20261023/final.pt 2>&1 | grep -v Warning
.venv/bin/python -u scripts/compare_value_nets.py checkpoints/search_finetune_20261019/final.pt checkpoints/searchvalue_mix05_20261022/final.pt checkpoints/scorehead_20261023/final.pt 2>&1 | grep -v Warning
log "all done"
