#!/bin/zsh
# Round 4: train (A) round-4 only and (B) round-4 + round-3 value/ownership, both from the round-3 model;
# then A vs B, the winner vs round 3, and T5. 400 visits, 100 games each.
cd ~/alphago-implementation
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.3 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.25
log() { echo "$(date '+%m-%d %H:%M') $*"; }
E=runs/experiments
R4=($E/selfplay_round4_800v_p0625_20000_seed20407102 $E/selfplay_round4_800v_p0625_20000_seed20437102)
R3=($E/selfplay_round3_800v_p0625_20000_20261030 $E/selfplay_round3_800v_p0625_20000_seed20297031 $E/selfplay_round3_800v_3000_20261020)
COMMON=(--init checkpoints/round3_merged_20261102/final.pt --dedup --value-fraction 1.0 --batch 64 --human-batch 16 --gen-batch 4 --epochs 2 --eval-every 2000 --save-evals)
ab() { .venv/bin/python -u scripts/match_parallel.py checkpoints/$1/final.pt checkpoints/$2/final.pt --visits 400 --pairs 50 --seed $4 --name $3 > /dev/null 2>&1
       python3 -c "import json;s=json.load(open('runs/matches/$3/summary.json'))['summary'];print('  $3',s['A_wins'],'/',s['games'],s['ci95'])"; }
log "[1/5] train A: round-4 only"
[ -f checkpoints/round4_only/final.pt ] || caffeinate -i .venv/bin/python -u scripts/train_search.py --selfplay $R4 --run round4_only $COMMON > runs/round4_only_train.log 2>&1 || { log "A failed"; exit 1; }
log "[2/5] train B: round-4 policy+value, round-3 value+ownership only"
[ -f checkpoints/round4_pool/final.pt ] || caffeinate -i .venv/bin/python -u scripts/train_search.py --selfplay $R4 --value-only $R3 --run round4_pool $COMMON > runs/round4_pool_train.log 2>&1 || { log "B failed"; exit 1; }
log "[3/5] A/B: pool (B) vs only (A)"
ab round4_pool round4_only ab_r4pool_vs_r4only_400 20261040
W=$(python3 -c "import json;print(json.load(open('runs/matches/ab_r4pool_vs_r4only_400/summary.json'))['summary']['A_win_rate'])")
BEST=$(python3 -c "print('round4_pool' if $W>=0.5 else 'round4_only')"); log "better round-4 model: $BEST"
log "[4/5] $BEST vs round 3"
ab $BEST round3_merged_20261102 ab_r4best_vs_r3_400 20261041
log "[5/5] T5"
.venv/bin/python -u scripts/t5_check.py checkpoints/round3_merged_20261102/final.pt checkpoints/round4_only/final.pt checkpoints/round4_pool/final.pt 2>&1 | grep -v Warning
log "all done"
