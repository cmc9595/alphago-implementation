#!/bin/zsh
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
log "waiting for shared training pipeline"
while pgrep -f value_then_shared.sh >/dev/null; do sleep 30; done
cat runs/compare_value_nets_2.txt
log "[1/4] separate: SL+value_gen_only vs SL+value_own, 400 visits, 100 games"
.venv/bin/python -u scripts/match_parallel.py checkpoints/sl_6x96/final.pt+checkpoints/value_gen_only/best.pt checkpoints/sl_6x96/final.pt+checkpoints/value_own/best.pt \
  --visits 400 --pairs 50 --seed 20261026 --name ab_valuegen_vs_valueown_400 | tail -3
log "[2/4] shared: shared_gen2 vs round2, 400 visits, 100 games"
.venv/bin/python -u scripts/match_parallel.py checkpoints/shared_gen2_20261022/final.pt checkpoints/search_round2_20261020/final.pt \
  --visits 400 --pairs 50 --seed 20261027 --name ab_sharedgen2_vs_round2_400 | tail -3
log "[3/4] T5 fixed test"
.venv/bin/python -u scripts/t5_check.py checkpoints/sl_6x96/final.pt+checkpoints/value_own/best.pt checkpoints/sl_6x96/final.pt+checkpoints/value_gen_only/best.pt \
  checkpoints/search_round2_20261020/final.pt checkpoints/shared_gen2_20261022/final.pt
read BEST <<< "$(python3 - <<'P'
import json
a=json.load(open('runs/matches/ab_valuegen_vs_valueown_400/summary.json'))['summary']['A_win_rate']
b=json.load(open('runs/matches/ab_sharedgen2_vs_round2_400/summary.json'))['summary']['A_win_rate']
sep='checkpoints/sl_6x96/final.pt:800:checkpoints/value_gen_only/best.pt' if a>=0.5 else 'checkpoints/sl_6x96/final.pt:800:checkpoints/value_own/best.pt'
print(sep)
P
)"
log "[4/4] rating ladder with $BEST (800 visits) vs KataGo human profiles, 40 games each"
K=$HOME/src/KataGo/cpp/build/katago; M=data/katago/b18c384nbt-humanv0.bin.gz
for r in 5k 1k 1d 3d; do
  .venv/bin/python scripts/match.py "mcts:$BEST:batch=32" \
    "gtp:$K gtp -model $M -human-model $M -config data/katago/human.cfg -override-config humanSLProfile=rank_$r" \
    --pairs 20 --parallel 2 --seed 20261028 --resume --name rating_${BEST:t:h:t}_vs_katago_rank_$r | tail -1
  python3 -c "import json;s=json.load(open('runs/matches/rating_${BEST:t:h:t}_vs_katago_rank_$r/summary.json'))['summary'];print('rank_$r',s['A_wins'],'/',s['games'],s['ci95'])"
done
log "all done"
