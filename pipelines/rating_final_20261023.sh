#!/bin/zsh
# Final KataGo human-profile ladder: finish the main engine, then the shared engine at the informative ranks.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
K=$HOME/src/KataGo/cpp/build/katago; M=data/katago/b18c384nbt-humanv0.bin.gz
run() { # name spec rank
  log "[$1] vs rank_$3"
  .venv/bin/python scripts/match.py "$2" \
    "gtp:$K gtp -model $M -human-model $M -config data/katago/human.cfg -override-config humanSLProfile=rank_$3" \
    --pairs 20 --parallel 4 --seed 20261028 --resume --name rating_$1_rank_$3 > /dev/null 2>&1
  python3 -c "import json;s=json.load(open('runs/matches/rating_$1_rank_$3/summary.json'))['summary'];print('  ',s['A_wins'],'/',s['games'],'win',s['A_win_rate'],'ci',s['ci95'])"
}
SEP="mcts:checkpoints/sl_6x96/final.pt:800:checkpoints/value_own/best.pt:batch=32"
SH="mcts:checkpoints/search_finetune_20261019/final.pt:800:batch=32"
for r in 1k 1d 3d; do run sep_valueown800 $SEP $r; done
for r in 1d 3d; do run shared_round1_800 $SH $r; done
log "all done"
