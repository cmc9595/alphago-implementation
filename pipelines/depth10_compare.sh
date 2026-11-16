#!/bin/zsh
# Train 6/10-block controls on identical data, then compare fixed-holdout MSE and matches.
set -eu
cd "${0:A:h}/.."
mkdir -p runs
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.12 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.10
E=runs/experiments
D=runs/datasets/round6_fixed1000
BASE=checkpoints/round5/final.pt
INIT6=checkpoints/round5_gpool_6x96/init.pt
INIT10=checkpoints/round5_gpool_10x96/init.pt
R6=depth_gpool_6x96_20261117
R10=depth_gpool_10x96_20261117
# Self-play only: no human or old generated-value anchor data required.
for p in $BASE $D/manifest.json; do
  [[ -f $p ]] || { echo "Missing $p; training not started"; exit 1; }
done
for d in $D/train $E/selfplay_round5_1600v_p125_f200_seed20507105 $E/selfplay_round3_800v_p0625_20000_seed20297031 $E/selfplay_round4_800v_p0625_20000_seed20407102 $E/selfplay_round4_800v_p0625_20000_seed20437102; do
  [[ -d $d ]] || { echo "Missing $d"; exit 1; }
done
[[ -f $INIT6 ]] || .venv/bin/python scripts/expand_model.py --init $BASE --out $INIT6 --blocks 6 --global-pool
[[ -f $INIT10 ]] || .venv/bin/python scripts/expand_model.py --init $BASE --out $INIT10 --blocks 10 --global-pool
COMMON=(--selfplay $E/selfplay_round5_1600v_p125_f200_seed20507105 $D/train
  --value-only $E/selfplay_round3_800v_p0625_20000_seed20297031 $E/selfplay_round4_800v_p0625_20000_seed20407102 $E/selfplay_round4_800v_p0625_20000_seed20437102
  --exclude-manifest $D/manifest.json --dedup --search-mix 0 --value-fraction 0.5
  --batch 64 --human-batch 0 --gen-batch 0 --epochs 2 --lr 0.001 --eval-every 2000 --save-evals --seed 20261117)
# Separate logs; no auto-resume of interrupted SGD (optimizer is not checkpointed).
for run init in $R6 $INIT6 $R10 $INIT10; do
  if [[ -e checkpoints/$run || -e runs/experiments/$run ]]; then
    echo "Existing run $run; inspect before restarting"; exit 1
  fi
done
for run init in $R6 $INIT6 $R10 $INIT10; do
  .venv/bin/python -u scripts/train_search.py --init $init --run $run $COMMON > runs/${run}_train.log 2>&1 &
  child=$!
  echo $child > runs/${run}_train.pid
  wait $child
done
.venv/bin/python -u scripts/eval_fixed_value.py checkpoints/round3_merged_20261102/final.pt $BASE checkpoints/$R6/final.pt checkpoints/$R10/final.pt --dataset $D --out runs/depth_gpool_fixed_value.json
for run in $R6 $R10; do
  .venv/bin/python -u scripts/match_parallel.py checkpoints/$run/final.pt $BASE --opening-net $BASE --visits 400 --pairs 50 --seed 20261118 --name ${run}_vs_round5_400 > runs/${run}_match.log 2>&1
done
echo 'Done. Matches use equal visits, NOT equal wall-clock time. Run time-matched tests before deployment.'
