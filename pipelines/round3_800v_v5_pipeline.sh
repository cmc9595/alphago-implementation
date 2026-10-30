#!/bin/zsh
# Round 3: 800-visit training moves, resume the 116 games already made with the round-2 model,
# then train with value on 1/5 of positions (was 1/20), A/B vs round-2 and round-1, T5 + value table.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
log "[1/4] self-play 3000 games (800/100 visits, tree reuse, no resign), round-2 model, resuming"
caffeinate -i .venv/bin/python -u scripts/selfplay_parallel.py --run selfplay_round3_800v_3000_20261020 --games 3000 --seed 20266020 \
  --model checkpoints/search_round2_20261020/final.pt --reuse-tree --visits 800 > runs/selfplay_round3_800v_3000_20261020.log 2>&1 &
echo $! > runs/selfplay_round3_800v_3000_20261020.pid; wait
N=$(ls runs/experiments/selfplay_round3_800v_3000_20261020/game_*.json | wc -l); log "self-play: $N games"
[ $N -ge 2900 ] || { log "too few games, stopping"; exit 1; }
log "[2/4] train from round-2: policy all, value 1/5, ownership all, gen-batch 4"
.venv/bin/python -u scripts/train_search.py --selfplay runs/experiments/selfplay_round3_800v_3000_20261020 \
  --init checkpoints/search_round2_20261020/final.pt --run round3_800v_v5_20261024 \
  --value-fraction 0.2 --batch 64 --human-batch 16 --gen-batch 4 --epochs 2 --eval-every 1000 --save-evals \
  > runs/round3_800v_v5_train.log 2>&1 || { log "train failed"; exit 1; }
log "[3/4] A/B 400 visits, 100 games each"
for pair in "search_round2_20261020 ab_r3v5_vs_round2_400 20261032" "search_finetune_20261019 ab_r3v5_vs_round1_400 20261033"; do
  set -- ${=pair}
  .venv/bin/python -u scripts/match_parallel.py checkpoints/round3_800v_v5_20261024/final.pt checkpoints/$1/final.pt \
    --visits 400 --pairs 50 --seed $3 --name $2 > /dev/null 2>&1
  python3 -c "import json;s=json.load(open('runs/matches/$2/summary.json'))['summary'];print('  $2',s['A_wins'],'/',s['games'],s['ci95'])"
done
log "[4/4] T5 + value table"
.venv/bin/python -u scripts/t5_check.py checkpoints/search_round2_20261020/final.pt checkpoints/round3_800v_v5_20261024/final.pt 2>&1 | grep -v Warning
.venv/bin/python -u scripts/compare_value_nets.py checkpoints/search_round2_20261020/final.pt checkpoints/round3_800v_v5_20261024/final.pt 2>&1 | grep -v Warning
log "all done"
