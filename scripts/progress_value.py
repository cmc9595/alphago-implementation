"""One screen for the value stage: data generation, human ownership labels, value-net training.

    python scripts/progress_value.py            # once
    python scripts/progress_value.py --watch    # refresh every 30s
"""

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BARS = " ▁▂▃▄▅▆▇█"


def bar(frac, width=36):
    frac = max(0.0, min(1.0, frac))
    return "█" * int(frac * width) + "░" * (width - int(frac * width))


def spark(values, width=40):
    values = values[-width:]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    return "".join(BARS[int((v - lo) / (hi - lo + 1e-12) * (len(BARS) - 1))] for v in values)


def running(pattern):
    return subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0


def elapsed_seconds(pattern):
    """Elapsed time of the oldest process matching pattern (from ps etime)."""
    pids = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
    best = 0
    for pid in pids:
        et = subprocess.run(["ps", "-o", "etime=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if not et:
            continue
        days, _, rest = et.rpartition("-")
        parts = [int(x) for x in rest.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        best = max(best, (int(days) if days else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2])
    return best


def eta(seconds):
    return f"남은 {timedelta(seconds=int(seconds))} (ETA {datetime.now() + timedelta(seconds=seconds):%m-%d %H:%M})"


def section_gen():
    d = ROOT / "data/value/gen"
    total = 977
    done = len(list(d.glob("batch_*.npz")))
    recs = [json.loads(l) for l in (d / "log.jsonl").read_text().splitlines()] if (d / "log.jsonl").exists() else []
    state = "RUNNING" if running("gen_value_data.py") else ("DONE" if done >= total else "STOPPED")
    lines = [f"[1] 가치망 데이터 생성 (SL 포석 → 무작위 1수 → RL 끝까지 → GNU Go)  [{state}]",
             f"    {bar(done / total)} {done}/{total} 묶음 ({done / total:.1%})"]
    if recs:
        sec = sum(r["sec_total"] for r in recs[-20:]) / len(recs[-20:])
        kept = sum(r["games_kept"] for r in recs)
        lines.append(f"    {sec:.0f}초/묶음   판정 완료 {kept:,}판 (목표 약 25만)   " + (eta((total - done) * sec) if state == "RUNNING" else ""))
        lines.append(f"    묶음당 시간 추이 {spark([r['sec_total'] for r in recs])}")
    return lines


def section_human():
    log = ROOT / "data/value/human_kgs.log"
    out = ROOT / "data/value/human_kgs.npz"
    state = "RUNNING" if running("prepare_human_ownership.py") else ("DONE" if out.exists() else "STOPPED")
    lines = [f"[2] KGS 사람 대국 소유권 (GNU Go 판정 = 기록 결과인 판만)  [{state}]"]
    text = log.read_text() if log.exists() else ""
    m = re.findall(r"judged (\d+)/(\d+), kept (\d+)", text)
    if m:
        judged, total, kept = map(int, m[-1])
        lines.append(f"    {bar(judged / total)} {judged:,}/{total:,}판 ({judged / total:.1%})   "
                     f"채택 {kept:,}판 ({kept / max(judged, 1):.0%})")
        if state == "RUNNING":
            el = elapsed_seconds("prepare_human_ownership.py") - 60  # ~1 min of SGF parsing first
            if judged > 0 and el > 0:
                lines.append("    " + eta((total - judged) * el / judged))
    fin = re.findall(r"kept (\d+)/(\d+) games \((\d+)%\)", text)
    if fin:
        lines.append(f"    완료: {int(fin[-1][0]):,}판 채택 ({fin[-1][2]}%)")
    return lines


def section_train(run):
    d = ROOT / "checkpoints" / run
    path = d / "metrics.jsonl"
    label = {"value_own": "가치 + 소유권 head (KataGo 방식)", "value_plain": "가치만 (AlphaGo 방식)"}.get(run, run)
    if not path.exists():
        return [f"[{run}] {label}  [대기 중]"]
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    total = 30_000
    state = "RUNNING" if running(f"train_value.py --run {run}") else ("DONE" if (d / "final.pt").exists() else "STOPPED")
    last = recs[-1]
    lines = [f"[{run}] {label}  [{state}]",
             f"    {bar(last['step'] / total)} {last['step']:,}/{total:,} 스텝   {last['samples_per_s']}/s   "
             + (eta((total - last['step']) * 256 / max(last['samples_per_s'], 1)) if state == "RUNNING" else ""),
             f"    학습 가치 MSE {last['train_value_mse']:.3f}" +
             (f"   학습 소유권 MSE {last['train_ownership_mse']:.3f}" if "train_ownership_mse" in last else "")]
    tests = [r for r in recs if "test" in r]
    if tests:
        t = tests[-1]["test"]
        ph = "  ".join(f"{k} {t[k]['mse']:.2f}" for k in ("opening", "early", "middle", "late") if k in t)
        lines.append(f"    테스트 MSE 전체 {t['all']['mse']:.3f} (승패 적중 {t['all']['winner_acc']:.0%})"
                     + (f"   생성 데이터 {t['gen_test']['mse']:.3f}" if "gen_test" in t else "")
                     + (f"   소유권 점별 정확도 {t['ownership_point_acc']:.1%}" if "ownership_point_acc" in t else ""))
        lines.append(f"    단계별 MSE: {ph}")
        lines.append(f"    테스트 MSE 추이 {spark([r['test']['all']['mse'] for r in tests])}  (AlphaGo 0.234, 목표 0.30 이하)")
    return lines


def render():
    out = []
    for sec in (section_gen(), section_human(), ["[3] 가치망 학습 (데이터가 준비되면 자동 시작)"],
                section_train("value_own"), section_train("value_plain")):
        out += sec + [""]
    log = ROOT / "runs/value_pipeline.log"
    if log.exists() and log.read_text().strip():
        out.append("파이프라인: " + log.read_text().strip().splitlines()[-1])
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    args = ap.parse_args()
    while True:
        text = render()
        if args.watch:
            print("\033[2J\033[H" + datetime.now().strftime("%H:%M:%S") + "\n" + text, flush=True)
            time.sleep(args.interval)
        else:
            print(text)
            break


if __name__ == "__main__":
    main()
