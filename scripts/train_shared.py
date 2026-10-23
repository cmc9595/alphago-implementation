"""Shared 6x96 policy/value/ownership model using existing human and generated data.

Human rows supervise policy only. Generated rows supervise value and ownership
only. No imitation of the RL moves, no MCTS labels and no teacher distillation.
"""
import os
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '.12')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO', '.10')
import argparse
from datetime import datetime, timedelta
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from alphago.data import GameData
from alphago.models import GoNet, NetConfig, pick_device
from alphago.gtp import load_net
from alphago.shared_training import supervised_losses, training_mode
from alphago.value_data import ValueData, load_gen
from value_source_ab import write_json, memory_guard, digest, duration

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = 'shared_supervised_6x96_20261018'


@torch.no_grad()
def evaluate(policy_net, value_net, evaluation, device, batch=32):
    policy_net.eval(); value_net.eval()
    hx, hy = evaluation['human_features'], evaluation['human_targets']
    correct, ce = 0, 0.
    for i in range(0, len(hy), batch):
        p, _ = policy_net(torch.from_numpy(hx[i:i+batch]).to(device).float())
        y = torch.from_numpy(hy[i:i+batch]).to(device)
        correct += int((p.argmax(1) == y).sum())
        ce += float(F.cross_entropy(p, y, reduction='sum'))
    gx, gz, go, plies = (evaluation[k] for k in ('gen_features','gen_values','gen_ownership','gen_plies'))
    predictions, own_squared, own_count = [], 0., 0
    for i in range(0, len(gz), batch):
        _, v, own = value_net.forward_all(torch.from_numpy(gx[i:i+batch]).to(device).float())
        predictions.append(v.cpu().numpy())
        if own is not None:
            target = torch.from_numpy(go[i:i+batch]).to(device)
            own_squared += float(((own-target)**2).sum()); own_count += target.numel()
    prediction = np.concatenate(predictions)
    stats = {'policy_acc': correct/len(hy), 'policy_ce': ce/len(hy), 'human_positions': len(hy),
             'value': {}}
    for lo, hi, name in [(0, 10000, 'all'), (0,30,'opening'), (30,80,'early'), (80,150,'middle'), (150,10000,'late')]:
        mask = (gz != 0) & (plies >= lo) & (plies < hi)
        if mask.any():
            stats['value'][name] = {'n': int(mask.sum()), 'mse': float(((prediction[mask]-gz[mask])**2).mean()),
                                   'winner_acc': float((np.sign(prediction[mask]) == gz[mask]).mean())}
    if own_count: stats['ownership_mse'] = own_squared/own_count
    return stats


def selection_score(metrics, baseline):
    # Keep both tasks visible; policy deterioration beyond 0.5 percentage points
    # receives an additional penalty. A minimum score does not establish strength.
    return (metrics['policy_ce']/baseline['policy_ce'] +
            metrics['value']['all']['mse']/baseline['value']['all']['mse'] +
            20*max(0., baseline['policy_acc']-.005-metrics['policy_acc']))


def progress(run, watch):
    out = ROOT/'runs/experiments'/run
    while True:
        if watch: print('\033[2J\033[H', end='')
        pipeline = json.loads((out/'pipeline.json').read_text()) if (out/'pipeline.json').exists() else {}
        state = json.loads((out/'state.json').read_text()) if (out/'state.json').exists() else {}
        print('사람 정책 + 생성 승패 · 공유 모델 학습')
        print('전체 단계:', pipeline.get('status',state.get('status','대기')))
        if state:
            done, total = state.get('step',0), state.get('steps',30000)
            width = int(30*min(1,done/total))
            print(f'[{"█"*width}{"░"*(30-width)}] {done:,}/{total:,} 스텝 · {state.get("status")}')
            rate = state.get('updates_per_second')
            if rate and state.get('status') == 'training':
                seconds = (total-done)/rate
                print('예상 남은 학습',duration(seconds),'· 종료',(datetime.now()+timedelta(seconds=seconds)).strftime('%m-%d %H:%M'))
                print('검증·후속 속도 측정·대국 평가 시간은 별도')
            if state.get('validation'):
                v = state['validation']
                print(f'검증 정책 {v["policy_acc"]:.2%} · 생성 가치 MSE {v["value"]["all"]["mse"]:.4f}')
            if state.get('error'): print('오류:',state['error'])
        match_dir = ROOT/'runs/matches'/f'{run}_vs_separate_400'
        if (match_dir/'summary.json').exists():
            summary = json.loads((match_dir/'summary.json').read_text())['summary']
            print('공유 vs 기존 분리망 평가:',summary.get('completed_games'),'/20판 · 공유망 성적',summary.get('A_win_rate'))
        print('현재 엔진에는 자동 적용하지 않습니다.')
        if not watch: break
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',default=DEFAULT_RUN)
    ap.add_argument('--steps',type=int,default=30000)
    ap.add_argument('--head-warmup',type=int,default=500)
    ap.add_argument('--batch-per-source',type=int,default=32)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--head-lr',type=float,default=.003)
    ap.add_argument('--ownership-weight',type=float,default=.5)
    ap.add_argument('--eval-every',type=int,default=1000)
    ap.add_argument('--eval-positions',type=int,default=8192)
    ap.add_argument('--seed',type=int,default=20261018)
    ap.add_argument('--watch',action='store_true')
    ap.add_argument('--progress',action='store_true')
    args=ap.parse_args();os.chdir(ROOT)
    if args.watch or args.progress:
        progress(args.run,args.watch);return
    if min(args.steps,args.batch_per_source,args.eval_every,args.eval_positions)<1:
        ap.error('Training budgets must be positive')
    out=ROOT/'runs/experiments'/args.run;out.mkdir(parents=True,exist_ok=True)
    ckdir=ROOT/'checkpoints'/args.run;ckdir.mkdir(parents=True,exist_ok=True)
    lock=open(out/'trainer.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'loading_data','step':0,'steps':args.steps,'pid':os.getpid()}
    def status(**changes):
        state.update(changes,updated=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(out/'state.json',state)
    status()
    try:
        torch.set_num_threads(4);torch.set_num_interop_threads(2)
        device=pick_device()
        if device.type!='mps':raise RuntimeError('MPS unavailable; refusing accidental CPU training')
        torch.mps.set_per_process_memory_fraction(.12)
        memory_guard()
        human_paths=[ROOT/'data/kgs.npz',ROOT/'data/fox_dan.npz']
        human=GameData(human_paths)  # Preserve the original SL whole-game split/seed.
        data=ValueData([load_gen()]) # Preserve the original gen whole-game split/seed.
        init_path=ROOT/'checkpoints/sl_6x96/final.pt'
        files=human_paths+sorted((ROOT/'data/value/gen').glob('batch_*.npz'))
        manifest={'args':vars(args),'init_sha256':digest(init_path),
                  'reference_value_sha256':digest(ROOT/'checkpoints/value_own/best.pt'),
                  'source_sha256':{name:digest(ROOT/name) for name in ('scripts/train_shared.py','alphago/shared_training.py')},
                  'data_files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in files],
                  'human_train_positions':human.num_positions(),'gen_train_positions':data.size(),
                  'human_policy_only':True,'generated_value_ownership_only':True,'bn_running_stats_frozen':True,
                  'validation_note':'Existing whole-game held-out splits reused; not a new untouched final test',
                  'teacher_distillation':False,'automatic_promotion':False}
        mf=out/'manifest.json'
        if mf.exists() and json.loads(mf.read_text())!=manifest:raise ValueError('Resume settings/data changed')
        write_json(mf,manifest)
        ep=out/'evaluation.npz'
        if ep.exists():
            with np.load(ep) as d:evaluation={k:d[k].copy() for k in d.files}
        else:
            hx,hy,_=human.sample(args.eval_positions,np.random.default_rng(71),'val',augment=False,threads=4)
            rows=data.index['test'][np.random.default_rng(72).permutation(data.size('test'))[:args.eval_positions]]
            gx,gz,go=data.batch_from_index(rows)
            evaluation=dict(human_features=hx,human_targets=hy,gen_features=gx,gen_values=gz,gen_ownership=go,gen_plies=rows[:,2],gen_rows=rows)
            np.savez_compressed(ep,**evaluation)
        memory_guard()
        init=torch.load(init_path,map_location='cpu',weights_only=False)
        torch.manual_seed(args.seed)
        cfg=NetConfig(**{**init['config'],'policy':True,'value':True,'ownership':True})
        net=GoNet(cfg).to(device)
        missing,unexpected=net.load_state_dict(init['model'],strict=False)
        if set(missing)!={'ownership_conv.weight','ownership_conv.bias'} or unexpected:
            raise ValueError(f'Unexpected init mismatch: {missing}, {unexpected}')
        baseline_path=out/'baseline.json'
        if baseline_path.exists():baseline=json.loads(baseline_path.read_text())
        else:
            status(status='baseline_evaluation')
            old_value=load_net(str(ROOT/'checkpoints/value_own/best.pt'),device)
            baseline=evaluate(net,old_value,evaluation,device,args.batch_per_source)
            write_json(baseline_path,baseline)
            write_json(out/'initial_shared.json',evaluate(net,net,evaluation,device,args.batch_per_source))
            del old_value;torch.mps.empty_cache()
        groups=[]
        for heads in (False,True):
            for decay in (False,True):
                params=[p for name,p in net.named_parameters() if name.startswith(('value_','ownership_'))==heads and (p.ndim>1)==decay]
                groups.append({'params':params,'lr':args.head_lr if heads else args.lr,'initial_lr':args.head_lr if heads else args.lr,'weight_decay':1e-4 if decay else 0.})
        opt=torch.optim.SGD(groups,momentum=.9,nesterov=True)
        rng=np.random.default_rng(args.seed);step=0;best=float('inf');last=ckdir/'last.pt'
        if last.exists():
            ck=torch.load(last,map_location='cpu',weights_only=False)
            net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
            step,best=ck['step'],ck['best_score'];rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'])
        def save(path,validation=None):
            tmp=path.with_suffix('.tmp')
            torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'config':vars(cfg),'args':vars(args),'step':step,
                        'best_score':best,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                        'validation':validation,'training':'shared backbone, human CE + generated outcome/ownership'},tmp)
            tmp.replace(path)
        status(status='training',step=step)
        count=0;totals={};start=time.monotonic()
        while step<args.steps:
            heads_only=step<min(args.head_warmup,args.steps)
            training_mode(net,heads_only)
            factor=.1+.9*.5*(1+math.cos(math.pi*max(0,step-args.head_warmup)/max(1,args.steps-args.head_warmup)))
            for group in opt.param_groups:group['lr']=group['initial_lr']*factor
            hx,hy,_=human.sample(args.batch_per_source,rng,augment=True,threads=4)
            gx,gz,go=data.sample(args.batch_per_source,rng)
            x=torch.from_numpy(np.concatenate([hx,gx])).to(device).float()
            loss,metrics=supervised_losses(net.forward_all(x),torch.from_numpy(hy).to(device),
                                          torch.from_numpy(gz).to(device),torch.from_numpy(go).to(device),args.ownership_weight)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite shared loss')
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True)
            opt.step();step+=1;count+=1
            for k,v in metrics.items():totals[k]=totals.get(k,0.)+float(v.detach())
            del loss,metrics,x
            if step%100==0 or step==args.steps:
                rec={'step':step,'heads_only':heads_only,**{k:v/count for k,v in totals.items()},
                     'updates_per_second':count/(time.monotonic()-start),'grad_norm_before_clip':float(norm)}
                if step%args.eval_every==0 or step==args.steps:
                    rec['validation']=evaluate(net,net,evaluation,device,args.batch_per_source)
                    score=selection_score(rec['validation'],baseline)
                    if score<best:
                        best=score;save(ckdir/'best.pt',rec['validation'])
                save(last,rec.get('validation'))
                with open(ckdir/'metrics.jsonl','a') as f:f.write(json.dumps(rec)+'\n')
                status(status='training',**rec);print(json.dumps(rec),flush=True)
                memory_guard();count=0;totals={};start=time.monotonic()
        save(ckdir/'final.pt',state.get('validation'))
        status(status='completed',step=step)
    except BaseException as exc:
        state.update(status='stopped' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));status();raise


if __name__=='__main__':main()
