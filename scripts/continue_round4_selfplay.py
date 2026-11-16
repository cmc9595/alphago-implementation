"""Wait for the current round-4 run, then generate additional 20k-game batches."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
INITIAL = 'selfplay_round4_800v_p0625_20000_seed20407102'


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    os.chdir(ROOT)
    pid = int((ROOT / 'runs' / f'{INITIAL}.pid').read_text())
    print(f'Waiting for {INITIAL}, PID {pid}', flush=True)
    while alive(pid):
        time.sleep(30)
    summary = json.loads((ROOT / 'runs/experiments' / INITIAL / 'summary.json').read_text())
    if summary['games'] != 20000:
        raise RuntimeError('Initial run did not finish; refusing to silently skip it')
    env = os.environ.copy()
    env.update(PYTORCH_MPS_HIGH_WATERMARK_RATIO='0.3', PYTORCH_MPS_LOW_WATERMARK_RATIO='0.25')
    batch = 1
    while True:
        free = shutil.disk_usage(ROOT).free
        if free < 10 * 1024**3:
            raise RuntimeError('Less than 10 GiB disk space free; stopping generation')
        seed = 20407102 + 30000 * batch
        run = f'selfplay_round4_800v_p0625_20000_seed{seed}'
        # Never overwrite an earlier run, including an interrupted one.
        if (ROOT / 'runs/experiments' / run).exists():
            raise RuntimeError(f'Run already exists: {run}; inspect before resuming')
        command = [sys.executable, '-u', 'scripts/selfplay_parallel.py', '--run', run,
                   '--games', '20000', '--seed', str(seed), '--model',
                   'checkpoints/round3_merged_20261102/final.pt', '--reuse-tree',
                   '--visits', '800', '--full-prob', '0.0625', '--fast-opening-sample']
        print(f'Starting {run}; free disk {free / 1024**3:.1f} GiB', flush=True)
        with (ROOT / 'runs' / f'{run}.log').open('ab') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            (ROOT / 'runs' / f'{run}.pid').write_text(str(child.pid) + '\n')
            (ROOT / 'runs/round4_continuous_current.txt').write_text(run + '\n')
            try:
                while child.poll() is None:
                    if shutil.disk_usage(ROOT).free < 10 * 1024**3:
                        raise RuntimeError('Less than 10 GiB disk space free')
                    time.sleep(30)
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
            if child.returncode:
                raise RuntimeError(f'{run} exited with code {child.returncode}; see its log')
        print(f'Completed {run}', flush=True)
        batch += 1


if __name__ == '__main__':
    main()
