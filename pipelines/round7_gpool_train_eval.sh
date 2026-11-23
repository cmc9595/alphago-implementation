#!/bin/zsh
# Fine-tune the 6-block global-pooling champion on round-7 self-play only.
set -eu
cd "${0:A:h}/.."
mkdir -p runs
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.12 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.10
R=round7_gpool6_20261120
SP=runs/experiments/selfplay_round7_gpool6_800v_p03_f200_30000_seed20730000
BASE=checkpoints/depth_gpool_6x96_20261117/final.pt
D=runs/datasets/round6_fixed1000
for p in $BASE $SP/summary.json $D/manifest.json; do
  [[ -f $p ]] || { echo "Missing $p"; exit 1; }
done
if [[ -e checkpoints/$R || -e runs/experiments/$R ]]; then
  echo "Existing run $R; inspect before restarting (no optimizer resume)"; exit 1
fi
echo '[1/4] train: round-7 only, outcome labels, no anchor data'
.venv/bin/python -u scripts/train_search.py --selfplay $SP --init $BASE --run $R \
  --exclude-manifest $D/manifest.json --dedup --search-mix 0 --value-fraction 0.5 \
  --batch 64 --human-batch 0 --gen-batch 0 --epochs 2 --lr 0.001 \
  --eval-every 1000 --save-evals --seed 20261120 > runs/${R}_train.log 2>&1 &
child=$!
echo $child > runs/${R}_train.pid
wait $child
echo '[2/4] fixed 1000-game value evaluation'
.venv/bin/python -u scripts/eval_fixed_value.py checkpoints/round5/final.pt $BASE checkpoints/$R/final.pt \
  --dataset $D --out runs/${R}_fixed_value.json
echo '[3/4] candidate vs parent: 400 visits, 100 games'
.venv/bin/python -u scripts/match_parallel.py checkpoints/$R/final.pt $BASE --opening-net $BASE \
  --visits 400 --pairs 50 --seed 20261121 --name ${R}_vs_parent_400 > runs/${R}_vs_parent.log 2>&1
echo '[4/4] candidate vs round5: 400 visits, 100 games'
.venv/bin/python -u scripts/match_parallel.py checkpoints/$R/final.pt checkpoints/round5/final.pt --opening-net $BASE \
  --visits 400 --pairs 50 --seed 20261122 --name ${R}_vs_round5_400 > runs/${R}_vs_round5.log 2>&1
echo 'Done. Equal visits, not equal wall-clock time. Browser model unchanged.'
