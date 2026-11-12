#!/bin/zsh
# Round 5 data, same training as round5 but value target = 0.5*outcome + 0.5*1600-visit root Q (--search-mix 0.5).
# Then A/B against round5 (outcome-only) and T5.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
R5=runs/experiments/selfplay_round5_1600v_p125_f200_seed20507105
log "[1/3] train"
[ -f checkpoints/round5_mix05/final.pt ] || caffeinate -i .venv/bin/python -u scripts/train_search.py --selfplay $R5 \
  --init checkpoints/round3_merged_20261102/final.pt --run round5_mix05 --dedup --search-mix 0.5 \
  --value-fraction 0.5 --batch 64 --human-batch 16 --gen-batch 4 --epochs 2 --eval-every 2000 --save-evals \
  > runs/round5_mix05_train.log 2>&1 || { log "train failed"; exit 1; }
log "[2/3] mix05 vs round5 (400 visits, 100 games)"
caffeinate -i .venv/bin/python -u scripts/match_parallel.py checkpoints/round5_mix05/final.pt checkpoints/round5/final.pt \
  --visits 400 --pairs 50 --seed 20261070 --name ab_r5mix_vs_r5_400 > /dev/null 2>&1
python3 -c "import json;s=json.load(open('runs/matches/ab_r5mix_vs_r5_400/summary.json'))['summary'];print('  mix05 vs r5',s['A_wins'],'/',s['games'],s['ci95'])"
log "[3/3] T5"
.venv/bin/python -u scripts/t5_check.py checkpoints/round5/final.pt checkpoints/round5_mix05/final.pt 2>&1 | grep -v Warning
log "all done"
