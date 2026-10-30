#!/bin/zsh
# After the 3,300-game self-play: fine-tune -> fixed-visit A/B match -> KataGo rating ladder.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
echo "$(date) waiting for fine-tune pid 86422"
while kill -0 86422 2>/dev/null; do sleep 20; done

echo "$(date) [1/3] fine-tune"
[ -f checkpoints/search_finetune_20261019/final.pt ] || .venv/bin/python -u scripts/train_search.py || exit 1

echo "$(date) [2/3] A/B: fine-tuned vs base shared, both 400 visits"
.venv/bin/python scripts/match.py \
  mcts:checkpoints/search_finetune_20261019/final.pt:400:batch=32 \
  mcts:checkpoints/shared_supervised_6x96_20261018/best.pt:400:batch=32 \
  --pairs 25 --parallel 2 --seed 20261021 --resume --name ab_search_finetune_vs_shared_400

echo "$(date) [3/3] speed benchmark (GPU free)"
SM=checkpoints/shared_supervised_6x96_20261018/best.pt
.venv/bin/python -u scripts/bench_eval_batch.py > runs/bench_eval_batch.log 2>&1
.venv/bin/python -u scripts/selfplay_parallel.py --run speed_parallel_basic --games 64 --model $SM
.venv/bin/python -u scripts/selfplay_parallel.py --run speed_parallel_reuse_resign --games 64 --model $SM --reuse-tree --resign
echo "$(date) all done"
