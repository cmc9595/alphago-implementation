#!/bin/zsh
# KataGo-style rounds: latest model always self-plays; stop only if new model wins < 50/100 vs previous.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
ab() { # new prev name seed
  .venv/bin/python -u scripts/match_parallel.py $1 $2 --visits 400 --pairs 50 --seed $4 --name $3
  python3 -c "import json;s=json.load(open('runs/matches/$3/summary.json'))['summary'];print(s['A_wins'],s['games'])"
}

log "waiting for round-2 self-play pid 94455"
while kill -0 94455 2>/dev/null; do sleep 60; done
N=$(ls runs/experiments/selfplay_round2_3000_20261019/game_*.json | wc -l); log "round-2 self-play: $N games"
[ $N -ge 2900 ] || { log "too few games, stopping"; exit 1; }

log "[R2 train] round1+round2 data, from round-1 model, gen-batch 8"
[ -f checkpoints/search_round2_20261020/final.pt ] || .venv/bin/python -u scripts/train_search.py \
  --selfplay runs/experiments/selfplay_cap_3300_20261019 runs/experiments/selfplay_round2_3000_20261019 \
  --init checkpoints/search_finetune_20261019/final.pt --run search_round2_20261020 --epochs 1.5 --gen-batch 8 || exit 1

log "[R2 match] round2 vs round1, 400 visits, 100 games"
read W G <<< "$(ab checkpoints/search_round2_20261020/final.pt checkpoints/search_finetune_20261019/final.pt ab_round2_vs_round1_400 20261022 | tail -1)"
log "round2 vs round1: $W / $G"
log "[R2 record] round2 vs original shared, 400 visits, 100 games (record only)"
ab checkpoints/search_round2_20261020/final.pt checkpoints/shared_supervised_6x96_20261018/best.pt ab_round2_vs_shared_400 20261023 | tail -1
python3 -c "import sys;sys.exit(0 if $W*2>=$G else 1)" || { log "round2 below 50%: stopping before round 3"; exit 0; }
[ $W -ge 60 ] && log "verdict: improvement confirmed (>=60)" || log "verdict: not confirmed (50-59), continuing with latest model"

R3=selfplay_round3_5000_20261020
log "[R3 self-play] 5000 games with round-2 model"
caffeinate -i .venv/bin/python -u scripts/selfplay_parallel.py --run $R3 --games 5000 --seed 20265020 \
  --model checkpoints/search_round2_20261020/final.pt --reuse-tree > runs/$R3.log 2>&1 &
echo $! > runs/$R3.pid; wait
N=$(ls runs/experiments/$R3/game_*.json | wc -l); log "round-3 self-play: $N games"
[ $N -ge 4800 ] || { log "too few games, stopping"; exit 1; }

log "[R3 train] round2+round3 data, from round-2 model, gen-batch 4"
[ -f checkpoints/search_round3_20261021/final.pt ] || .venv/bin/python -u scripts/train_search.py \
  --selfplay runs/experiments/selfplay_round2_3000_20261019 runs/experiments/$R3 \
  --init checkpoints/search_round2_20261020/final.pt --run search_round3_20261021 --epochs 1.5 --gen-batch 4 || exit 1

log "[R3 match] round3 vs round2, 400 visits, 100 games"
read W G <<< "$(ab checkpoints/search_round3_20261021/final.pt checkpoints/search_round2_20261020/final.pt ab_round3_vs_round2_400 20261024 | tail -1)"
log "round3 vs round2: $W / $G"
log "all done"
