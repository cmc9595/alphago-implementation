#!/bin/zsh
# Is the value net capacity-limited? Same data/eval rows as value_own (30k steps, test MSE 0.621 / gen 0.464).
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
echo "$(date '+%m-%d %H:%M') [A] 8x128 random init, 30k steps"
caffeinate -i .venv/bin/python -u scripts/train_value.py --run value_8x128 --init none --blocks 8 --channels 128 --steps 30000
echo "$(date '+%m-%d %H:%M') [B] 6x96 continue value_own/final 30k more steps (lr 0.005)"
caffeinate -i .venv/bin/python -u scripts/train_value.py --run value_own_cont --init checkpoints/value_own/final.pt --steps 30000 --lr 0.005 --warmup 200
echo "$(date '+%m-%d %H:%M') done"
