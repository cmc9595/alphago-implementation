#!/bin/zsh
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
SP=$(cat runs/selfplay_round3_800v_3000_20261020.pid); log "waiting for round-3 (800 visits) self-play pid $SP"
while kill -0 $SP 2>/dev/null; do sleep 60; done
N=$(ls runs/experiments/selfplay_round3_800v_3000_20261020/game_*.json | wc -l); log "round-3 self-play: $N games"
[ $N -ge 2900 ] || { log "too few games, stopping"; exit 1; }
log "[R3 train] round3 (800v) data only, from round-2 model, gen-batch 4"
[ -f checkpoints/search_round3_20261021/final.pt ] || .venv/bin/python -u scripts/train_search.py \
  --selfplay runs/experiments/selfplay_round3_800v_3000_20261020 --init checkpoints/search_round2_20261020/final.pt \
  --run search_round3_20261021 --epochs 2 --gen-batch 4 || exit 1
log "[R3 match] round3 vs round2, 400 visits, 100 games"
.venv/bin/python -u scripts/match_parallel.py checkpoints/search_round3_20261021/final.pt checkpoints/search_round2_20261020/final.pt \
  --visits 400 --pairs 50 --seed 20261024 --name ab_round3_vs_round2_400
log "[R3 record] round3 vs round1, 400 visits, 100 games"
.venv/bin/python -u scripts/match_parallel.py checkpoints/search_round3_20261021/final.pt checkpoints/search_finetune_20261019/final.pt \
  --visits 400 --pairs 50 --seed 20261025 --name ab_round3_vs_round1_400
log "all done"
