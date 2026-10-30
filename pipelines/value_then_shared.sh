#!/bin/zsh
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
log "waiting for value_gen_only training"
while pgrep -f "train_value.py --run value_gen_only" >/dev/null; do sleep 30; done
log "[1/3] compare value_own vs value_gen_only"
.venv/bin/python -u scripts/compare_value_nets.py checkpoints/value_own/best.pt checkpoints/value_gen_only/best.pt > runs/compare_value_nets_1.txt 2>&1
cat runs/compare_value_nets_1.txt
log "[2/3] shared model: round-2 init; per step 32 self-play (policy+own, value 1/20) + 32 human (policy) + 128 generated (value+own)"
caffeinate -i .venv/bin/python -u scripts/train_search.py \
  --selfplay runs/experiments/selfplay_cap_3300_20261019 runs/experiments/selfplay_round2_3000_20261019 \
  --init checkpoints/search_round2_20261020/final.pt --run shared_gen2_20261022 \
  --batch 32 --human-batch 32 --gen-batch 128 --gen-sources gen+gen2 --steps 30000 --eval-every 2000 --save-evals \
  > runs/shared_gen2_20261022.log 2>&1 || { log "shared training failed"; exit 1; }
log "[3/3] compare all three"
.venv/bin/python -u scripts/compare_value_nets.py checkpoints/value_own/best.pt checkpoints/value_gen_only/best.pt \
  checkpoints/search_round2_20261020/final.pt checkpoints/shared_gen2_20261022/final.pt > runs/compare_value_nets_2.txt 2>&1
cat runs/compare_value_nets_2.txt
log "all done"
