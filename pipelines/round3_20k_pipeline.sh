#!/bin/zsh
# Round 3: 20,000 games, 800-visit training moves on 6.25% of moves, round-2 model.
# Train on these + the 136 earlier 800-visit games (25%), value on all positions.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
log "[1/4] self-play 20000 games (800 visits on 6.25% of moves, 100 otherwise, tree reuse, no resign)"
caffeinate -i .venv/bin/python -u scripts/selfplay_parallel.py --run selfplay_round3_800v_p0625_20000_20261030 --games 20000 --seed 20267030 \
  --model checkpoints/search_round2_20261020/final.pt --reuse-tree --visits 800 --full-prob 0.0625 > runs/selfplay_round3_800v_p0625_20000_20261030.log 2>&1 &
echo $! > runs/selfplay_round3_800v_p0625_20000_20261030.pid; wait
N=$(ls runs/experiments/selfplay_round3_800v_p0625_20000_20261030/game_*.json | wc -l); log "self-play: $N games"
[ $N -ge 19000 ] || { log "too few games, stopping"; exit 1; }
log "[2/4] train from round-2: policy all, value ALL, ownership all, gen-batch 4"
.venv/bin/python -u scripts/train_search.py --selfplay runs/experiments/selfplay_round3_800v_p0625_20000_20261030 runs/experiments/selfplay_round3_800v_3000_20261020 \
  --init checkpoints/search_round2_20261020/final.pt --run round3_20k_v3_20261101 \
  --value-fraction 1.0 --batch 64 --human-batch 16 --gen-batch 4 --epochs 2 --eval-every 1000 --save-evals \
  > runs/round3_20k_v3_train.log 2>&1 || { log "train failed"; exit 1; }
log "[3/4] A/B 400 visits, 100 games each"
for pair in "search_round2_20261020 ab_r3_20k_vs_round2_400 20261034" "search_finetune_20261019 ab_r3_20k_vs_round1_400 20261035"; do
  set -- ${=pair}
  .venv/bin/python -u scripts/match_parallel.py checkpoints/round3_20k_v3_20261101/final.pt checkpoints/$1/final.pt \
    --visits 400 --pairs 50 --seed $3 --name $2 > /dev/null 2>&1
  python3 -c "import json;s=json.load(open('runs/matches/$2/summary.json'))['summary'];print('  $2',s['A_wins'],'/',s['games'],s['ci95'])"
done
log "[4/4] T5 + value table"
.venv/bin/python -u scripts/t5_check.py checkpoints/search_round2_20261020/final.pt checkpoints/round3_20k_v3_20261101/final.pt 2>&1 | grep -v Warning
.venv/bin/python -u scripts/compare_value_nets.py checkpoints/search_round2_20261020/final.pt checkpoints/round3_20k_v3_20261101/final.pt 2>&1 | grep -v Warning
log "all done"
