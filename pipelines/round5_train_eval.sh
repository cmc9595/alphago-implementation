#!/bin/zsh
# Round 5: train from round-3 model on 20,000 games (1,600-visit targets on 12.5% of moves, 200-visit fast moves).
# Policy + ownership on all training positions, value on half (~18 per game, as in round 3). Then A/B and T5.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
R5=runs/experiments/selfplay_round5_1600v_p125_f200_seed20507105
ab() { .venv/bin/python -u scripts/match_parallel.py checkpoints/round5/final.pt checkpoints/$1/final.pt --visits 400 --pairs 50 --seed $3 --name $2 > /dev/null 2>&1
       python3 -c "import json;s=json.load(open('runs/matches/$2/summary.json'))['summary'];print('  $2',s['A_wins'],'/',s['games'],s['ci95'])"; }
log "[1/4] train"
[ -f checkpoints/round5/final.pt ] || caffeinate -i .venv/bin/python -u scripts/train_search.py --selfplay $R5 \
  --init checkpoints/round3_merged_20261102/final.pt --run round5 --dedup \
  --value-fraction 0.5 --batch 64 --human-batch 16 --gen-batch 4 --epochs 2 --eval-every 2000 --save-evals \
  > runs/round5_train.log 2>&1 || { log "train failed"; exit 1; }
log "[2/4] round 5 vs round 3"; ab round3_merged_20261102 ab_r5_vs_r3_400 20261050
log "[3/4] round 5 vs round 1"; ab search_finetune_20261019 ab_r5_vs_r1_400 20261051
log "[4/4] T5"
.venv/bin/python -u scripts/t5_check.py checkpoints/round3_merged_20261102/final.pt checkpoints/round5/final.pt 2>&1 | grep -v Warning
log "all done"
