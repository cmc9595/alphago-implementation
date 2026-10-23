"""Training progress at a glance.

    python scripts/progress.py                 # once
    python scripts/progress.py --watch         # refresh every 30s (Ctrl+C to quit)
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

BARS = " ▁▂▃▄▅▆▇█"


def spark(values, width=40):
    values = values[-width:]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    return "".join(BARS[int((v - lo) / (hi - lo + 1e-12) * (len(BARS) - 1))] for v in values)


def render_rl(run_dir: Path, recs) -> str:
    last = recs[-1]
    total = 3000
    try:
        import torch

        ck = run_dir / "last.pt"
        total = torch.load(ck, map_location="cpu", weights_only=False)["args"]["iterations"] if ck.exists() else total
    except Exception:
        pass
    # Prefer the running command's --iterations (a resumed run may target more than last.pt says).
    ps = subprocess.run(["pgrep", "-fl", f"train_rl.py.*{run_dir.name}"], capture_output=True, text=True).stdout.split()
    if "--iterations" in ps:
        total = int(ps[ps.index("--iterations") + 1])
    start = recs[0]["iteration"] - 1 if run_dir.name.endswith("_cont") else 0
    it = last["iteration"]
    sec = sum(r["sec_total"] for r in recs[-20:]) / len(recs[-20:])
    remaining = (total - it) * sec
    running = subprocess.run(["pgrep", "-f", f"train_rl.py.*{run_dir.name}"], capture_output=True).returncode == 0
    pct = it / total
    new_evals = [r for r in recs if "win_rate_vs_sl" in r and r["iteration"] > 3000]
    bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
    games = sum(1 for _ in recs) * 256
    evals = [r for r in recs if "win_rate_vs_sl" in r]
    lines = [
        f"run {run_dir.name}  [{'RUNNING' if running else 'STOPPED'}]",
        f"{bar} iteration {it:,}/{total:,} ({pct:.1%})   self-play games ~{games:,}",
        f"{sec:.0f}s/iteration   remaining {timedelta(seconds=int(remaining))}  "
        f"(ETA {datetime.now() + timedelta(seconds=remaining):%m-%d %H:%M})",
        "",
        f"vs pool opponent: win {last['win_rate']:.1%} (opponent {last['opponent']})",
        (f"avg moves {last['avg_moves']:.0f}   계가 {last['referee']}   (참고) 사석 무시 계가였다면 바뀌었을 승패 {last['cleanup_flip_rate']:.0%}"
         if str(last.get("referee", "")).startswith("gnugo") else
         f"avg moves {last['avg_moves']:.0f} + cleanup {last['avg_cleanup']:.0f}   cleanup flips {last['cleanup_flip_rate']:.0%}"),
        f"value mse {last['value']:.3f}   policy entropy {last['entropy']:.2f}   grad norm {last['grad_norm']}   "
        f"KL to {last.get('kl_ref', 'sl')} {last.get('kl', 0):.3f}   learner pass {last.get('pass_rate', 0):.1%}   capped {last['capped']}",
        f"win rate vs pool trend  {spark([r['win_rate'] for r in recs], 50)}",
    ]
    if evals:
        lines += ["", "평가: RL vs 원래 SL (128판, 샘플링 대국)  " +
                  "  ".join(f"{r['iteration']}:{r['win_rate_vs_sl']:.0%}" + ("(복구)" if "rollback" in r else "")
                            for r in evals[-8:]),
                  "(참고: AlphaGo RL 정책망은 SL을 약 80% 이김)"]
    return "\n".join(lines)


def render(run_dir: Path) -> str:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return f"run {run_dir.name}: 아직 기록 없음 (시작 중)"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not recs:
        return "no metrics yet"
    if "iteration" in recs[-1]:
        return render_rl(run_dir, recs)
    last = recs[-1]
    ck = run_dir / "last.pt"
    total = 300_000
    try:
        import torch

        total = torch.load(ck, map_location="cpu", weights_only=False)["args"]["steps"] if ck.exists() else total
    except Exception:
        pass
    step = last["step"]
    speed = sum(r["samples_per_s"] for r in recs[-20:]) / len(recs[-20:])
    remaining = (total - step) * 256 / max(speed, 1)
    eta = datetime.now() + timedelta(seconds=remaining)
    running = subprocess.run(["pgrep", "-f", f"train_sl.py.*{run_dir.name}"], capture_output=True).returncode == 0
    pct = step / total
    bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))

    vals = [r for r in recs if "val_acc" in r]
    lines = [
        f"run {run_dir.name}  [{'RUNNING' if running else 'STOPPED'}]",
        f"{bar} {step:,}/{total:,} ({pct:.1%})",
        f"speed {speed:,.0f} samples/s   lr {last['lr']}   "
        f"remaining {timedelta(seconds=int(remaining))}  (ETA {eta:%m-%d %H:%M})",
        "",
        f"train acc   {last['train_acc']:.1%}   policy loss {last['policy_loss']:.3f}",
        f"train acc trend  {spark([r['train_acc'] for r in recs], 50)}",
    ]
    if vals:
        v = vals[-1]
        lines += [
            "",
            f"validation @ step {v['step']:,}: acc {v['val_acc']:.1%}   top5 {v['val_top5']:.1%}   "
            f"value mse {v['val_value_mse']:.3f}",
            "val acc history: " + "  ".join(f"{r['step'] // 1000}k:{r['val_acc']:.1%}" for r in vals[-8:]),
            "(참고: AlphaGo SL 정책망 KGS 57.0%. 수 예측 정확도는 기력이 아니다 — 실제 기력은 대국으로 측정)",
        ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="sl_6x96")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()
    run_dir = Path(__file__).resolve().parent.parent / "checkpoints" / args.run
    while True:
        out = render(run_dir)
        if args.watch:
            print("\033[2J\033[H" + datetime.now().strftime("%H:%M:%S") + "\n" + out, flush=True)
            time.sleep(args.interval)
        else:
            print(out)
            break


if __name__ == "__main__":
    main()
