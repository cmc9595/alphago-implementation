#!/bin/zsh
# Round 3 training on merged self-play: this Mac 9,293 games + M1 Max 20,000 games + 136 earlier 800-visit games.
# From round-2 model; policy all, value ALL, ownership all; exact-duplicate games dropped; gen-batch 4.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
E=runs/experiments
log "[1/3] train"
caffeinate -i .venv/bin/python -u scripts/train_search.py \
  --selfplay $E/selfplay_round3_800v_p0625_20000_20261030 $E/selfplay_round3_800v_p0625_20000_seed20297031 $E/selfplay_round3_800v_3000_20261020 \
  --init checkpoints/search_round2_20261020/final.pt --run round3_merged_20261102 --dedup \
  --value-fraction 1.0 --batch 64 --human-batch 16 --gen-batch 4 --epochs 2 --eval-every 2000 --save-evals \
  > runs/round3_merged_train.log 2>&1 || { log "train failed"; exit 1; }
log "[2/3] A/B 400 visits, 100 games each"
for pair in "search_round2_20261020 ab_r3m_vs_round2_400 20261036" "search_finetune_20261019 ab_r3m_vs_round1_400 20261037"; do
  set -- ${=pair}
  .venv/bin/python -u scripts/match_parallel.py checkpoints/round3_merged_20261102/final.pt checkpoints/$1/final.pt \
    --visits 400 --pairs 50 --seed $3 --name $2 > /dev/null 2>&1
  python3 -c "import json;s=json.load(open('runs/matches/$2/summary.json'))['summary'];print('  $2',s['A_wins'],'/',s['games'],s['ci95'])"
done
log "[3/3] T5 + value table"
.venv/bin/python -u scripts/t5_check.py checkpoints/search_round2_20261020/final.pt checkpoints/round3_merged_20261102/final.pt 2>&1 | grep -v Warning
.venv/bin/python -u scripts/compare_value_nets.py checkpoints/search_round2_20261020/final.pt checkpoints/round3_merged_20261102/final.pt 2>&1 | grep -v Warning
log "all done"
