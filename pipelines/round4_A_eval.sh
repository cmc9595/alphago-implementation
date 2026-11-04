#!/bin/zsh
# Round-4 model A (round-4 data only) vs round 3 and vs round 1, 400 visits, 100 games each; then T5.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
ab() { .venv/bin/python -u scripts/match_parallel.py checkpoints/round4_only/final.pt checkpoints/$1/final.pt --visits 400 --pairs 50 --seed $3 --name $2 > /dev/null 2>&1
       python3 -c "import json;s=json.load(open('runs/matches/$2/summary.json'))['summary'];print('  $2',s['A_wins'],'/',s['games'],s['ci95'])"; }
log "[1/3] A vs round 3"; ab round3_merged_20261102 ab_r4only_vs_r3_400 20261042
log "[2/3] A vs round 1"; ab search_finetune_20261019 ab_r4only_vs_r1_400 20261043
log "[3/3] T5"
.venv/bin/python -u scripts/t5_check.py checkpoints/round3_merged_20261102/final.pt checkpoints/round4_only/final.pt 2>&1 | grep -v Warning
log "all done"
