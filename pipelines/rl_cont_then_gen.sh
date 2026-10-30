#!/bin/zsh
# After capacity test: continue RL 600 more iterations (~2-3h), compare vs current RL (400 games),
# and if the new RL wins >= 55%, generate 1.5M AlphaGo-style value games with it.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
log "[1/3] waiting for RL training pid 63066 to finish"
while kill -0 63066 2>/dev/null; do sleep 30; done
[ -f checkpoints/rl_6x96_cont/final.pt ] || { log "RL final.pt missing"; exit 1; }
log "[2/3] new RL vs current RL, policy only, 400 games"
.venv/bin/python scripts/eval_policies.py checkpoints/rl_6x96_cont/final.pt checkpoints/rl_6x96/final.pt --games 400 --batch 200 \
  | tail -1 | tee runs/rl_cont_vs_rl.json
WR=$(python3 -c "import json;print(json.load(open('runs/rl_cont_vs_rl.json'))['A_win_rate'])")
log "new RL win rate vs current RL: $WR"
if python3 -c "import sys;sys.exit(0 if $WR>=0.55 else 1)"; then GEN_RL=checkpoints/rl_6x96_cont/final.pt; else GEN_RL=checkpoints/rl_6x96/final.pt; fi
log "[3/3] generate 1.5M value games with $GEN_RL"
caffeinate -i .venv/bin/python -u scripts/gen_value_data.py --rl $GEN_RL \
  --out data/value/gen_rlcont --games 1500000 --positions 2 --seed 20261020 > runs/gen_rlcont.log 2>&1
log "all done"
