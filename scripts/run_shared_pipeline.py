"""Train shared supervised model, benchmark it, then play 20 equal-400-visit games."""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','.10')
import fcntl,json,signal,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
RUN='shared_supervised_6x96_20261018'


def main():
    os.chdir(ROOT)
    out=ROOT/'runs/experiments'/RUN;out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'pipeline.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'status':'preparing'}
    def status(**changes):
        state.update(changes,updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        tmp=out/'pipeline.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2));tmp.replace(out/'pipeline.json')
    def stage(name,command):
        status(status=name)
        with open(out/f'{name}.log','a') as log:
            child=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status(child_pid=child.pid)
            try:
                code=child.wait()
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    child.wait(timeout=30)
                raise
        if code:raise RuntimeError(f'{name} exited with {code}; see {name}.log')
    try:
        # Existing runs use these locks. Avoid overlapping their GPU jobs on resume.
        locks=[open(ROOT/rel,'a') for rel in ['runs/experiments/value_source_ab_20261018/experiment.lock','runs/experiments/search_iteration_400_20261018/runner.lock']]
        while True:
            acquired=[]
            try:
                for f in locks:
                    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired.append(f)
                break
            except BlockingIOError:
                for f in acquired:fcntl.flock(f,fcntl.LOCK_UN)
                status(status='waiting_for_other_gpu_job');time.sleep(30)
        stage('training',['scripts/train_shared.py','--run',RUN])
        model=f'checkpoints/{RUN}/best.pt'
        if not (ROOT/model).exists():raise RuntimeError('No selected checkpoint; refusing evaluation')
        stage('benchmark',['scripts/benchmark_shared.py','--model',model,'--out',str(out/'speed.json')])
        stage('matches',['scripts/match.py',f'mcts:{model}:400:batch=32',
                         'mcts:checkpoints/sl_6x96/final.pt:400:checkpoints/value_own/best.pt:batch=32',
                         '--pairs','10','--parallel','1','--max-moves','600','--name',f'{RUN}_vs_separate_400','--resume'])
        status(status='completed',automatic_promotion=False)
    except BaseException as exc:
        status(status='failed',error=repr(exc));raise


if __name__=='__main__':main()
