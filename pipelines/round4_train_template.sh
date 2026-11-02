#!/bin/zsh
# Round-4 training template: policy from round 4 only; value + ownership from round 4 AND round 3 (800-visit era).
# Fill R4 with the round-4 self-play folder(s) once they exist under runs/experiments/.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
E=runs/experiments
R4=($E/selfplay_round4_800v_p0625_20000_seed20407102)
R3=($E/selfplay_round3_800v_p0625_20000_20261030 $E/selfplay_round3_800v_p0625_20000_seed20297031 $E/selfplay_round3_800v_3000_20261020)
caffeinate -i .venv/bin/python -u scripts/train_search.py --selfplay $R4 --value-only $R3 \
  --init checkpoints/round3_merged_20261102/final.pt --run round4_${1:-pool}_$(date +%Y%m%d) --dedup \
  --value-fraction 1.0 --batch 64 --human-batch 16 --gen-batch 4 --epochs 2 --eval-every 2000 --save-evals \
  > runs/round4_${1:-pool}_train.log 2>&1
