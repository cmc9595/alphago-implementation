#!/bin/zsh
# Round-3 shared model vs KataGo human-imitation 3d (and 5d if 3d >= 60%), 800 visits, 40 games each.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
K=$HOME/src/KataGo/cpp/build/katago; M=data/katago/b18c384nbt-humanv0.bin.gz
SPEC="mcts:checkpoints/round3_merged_20261102/final.pt:800:batch=32"
run() {
  log "[shared_round3_800] vs rank_$1"
  .venv/bin/python scripts/match.py "$SPEC" \
    "gtp:$K gtp -model $M -human-model $M -config data/katago/human.cfg -override-config humanSLProfile=rank_$1" \
    --pairs 20 --parallel 4 --seed 20261028 --resume --name rating_shared_round3_800_rank_$1 > /dev/null 2>&1
  python3 -c "import json;s=json.load(open('runs/matches/rating_shared_round3_800_rank_$1/summary.json'))['summary'];print('  ',s['A_wins'],'/',s['games'],'win',s['A_win_rate'],'ci',s['ci95'])"
}
run 3d
W=$(python3 -c "import json;print(json.load(open('runs/matches/rating_shared_round3_800_rank_3d/summary.json'))['summary']['A_win_rate'])")
python3 -c "import sys;sys.exit(0 if $W>=0.6 else 1)" && run 5d
log "all done"
