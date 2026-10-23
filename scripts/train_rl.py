"""RL stage: REINFORCE policy-gradient self-play against a pool of past versions (AlphaGo 2016).

    python scripts/train_rl.py --sl checkpoints/sl_6x96/final.pt
    python scripts/train_rl.py --sl ... --iterations 3 --games 16 --run rl_smoke     # quick check

Per iteration: play `--games` games (learner takes each colour half the time) against an
opponent drawn from the pool, then one SGD step on
    (1/games) * sum_t [ -(z_t - v(s_t)) * log pi(a_t | s_t) + beta * KL(pi(.|s_t) || pi_SL(.|s_t)) ]
over the learner's moves. The value head is a variance-reducing baseline regressed toward z on
detached trunk features. The KL anchor to the SL policy and the automatic rollback (revert to
the best checkpoint when the win rate vs SL drops) were added after a first run collapsed:
the policy stopped passing, filled its own territory and killed its own groups.
Results are judged by GNU Go (dead stones removed) by default; every iteration's games are
archived under checkpoints/<run>/games/ with GNU Go ownership for the value / ownership stage.
Resumes from checkpoints/<run>/last.pt.
"""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alphago import replay_features
from alphago.gnugo import Referee
from alphago.models import GoNet, NetConfig, pick_device
from alphago.selfplay import SelfPlayConfig, apply_referee, play_batch


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = GoNet(NetConfig(**ck["config"])).to(device)
    net.load_state_dict(ck["model"])
    return net, ck


def reinforce_step(net, kl_net, opt, recs, device, args, rng):
    """One policy-gradient update from a batch of finished games. Returns stats."""
    n = 19
    moves, starts, plies, actions, z = [], [], [], [], []
    base = 0
    for rec in recs:
        m = np.asarray(rec.moves, np.int16)
        moves.append(m)
        first = 0 if rec.learner_black else 1  # Black moves on even plies
        learner_plies = np.arange(first, len(m), 2)
        keep = learner_plies[rng.random(len(learner_plies)) < args.sample_fraction]
        starts.append(np.full(len(keep), base, np.int64))
        plies.append(keep.astype(np.int32))
        actions.append(m[keep].astype(np.int64))
        z.append(np.full(len(keep), 1.0 if rec.learner_won else -1.0, np.float32))
        base += len(m)
    moves = np.concatenate(moves)
    starts, plies = np.concatenate(starts), np.concatenate(plies)
    actions, z = np.concatenate(actions), np.concatenate(z)
    total = len(actions)
    # Unbiased estimate of the per-game sum over all learner moves despite subsampling.
    pg_scale = 1.0 / (len(recs) * args.sample_fraction)

    net.eval()  # keep BatchNorm statistics frozen; gradients still flow
    opt.zero_grad(set_to_none=True)
    sums = {"pg": 0.0, "kl": 0.0, "value": 0.0, "entropy": 0.0, "adv_abs": 0.0, "pass_rate": 0.0}
    for i in range(0, total, args.chunk):
        sl = slice(i, i + args.chunk)
        x = torch.from_numpy(replay_features(moves, starts[sl], plies[sl], n)).to(device).float()
        a = torch.from_numpy(actions[sl]).to(device)
        zt = torch.from_numpy(z[sl]).to(device)
        logits, v = net(x, detach_value_trunk=True)
        logp = F.log_softmax(logits, 1)
        with torch.no_grad():
            sl_logp = F.log_softmax(kl_net(x)[0], 1)
        chosen = logp.gather(1, a[:, None]).squeeze(1)
        adv = zt - v.detach() if args.baseline else zt
        pg = -(adv * chosen).sum() * pg_scale
        kl = (logp.exp() * (logp - sl_logp)).sum() * pg_scale  # KL(pi || pi_anchor), same per-game scaling
        vloss = F.mse_loss(v, zt, reduction="sum") / total
        (pg + args.kl_weight * kl + args.value_weight * vloss).backward()
        sums["pg"] += pg.item()
        sums["kl"] += kl.item() / (total * pg_scale)  # report mean KL per position
        sums["pass_rate"] += (a == 361).sum().item() / total
        sums["value"] += vloss.item()
        sums["entropy"] += -(logp.exp() * logp).sum(1).sum().item() / total
        sums["adv_abs"] += adv.abs().sum().item() / total
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip).item()
    opt.step()
    return {"positions": total, "grad_norm": round(grad_norm, 3), **{k: round(v, 4) for k, v in sums.items()}}


def save_games(path, recs):
    """Self-play archive for the value / ownership stage: moves, result, GNU Go ownership."""
    lengths = np.array([len(r.moves) for r in recs], np.int32)
    np.savez_compressed(
        path,
        moves=np.concatenate([np.asarray(r.moves, np.int16) for r in recs]),
        offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        score=np.array([r.score_after for r in recs], np.float32),
        judged=np.array([r.judged == "gnugo" for r in recs]),
        ownership=np.stack([r.ownership if r.ownership is not None else np.zeros(361, np.int8) for r in recs]),
    )


def evaluate(learner, reference, device, games, cfg, seed, referee):
    recs = play_batch(learner, reference, games, np.random.default_rng(seed), device, cfg, referee=referee)
    return sum(r.learner_won for r in recs) / len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sl", default="checkpoints/sl_6x96/final.pt")
    ap.add_argument("--kl-ref", default=None,
                    help="policy the KL term anchors to (default: --sl). Moving it to a recent RL snapshot "
                         "lets learning continue once KL to SL saturates (ProRL-style reference reset)")
    ap.add_argument("--run", default="rl_6x96")
    ap.add_argument("--iterations", type=int, default=4000)
    ap.add_argument("--games", type=int, default=256, help="games per iteration")
    ap.add_argument("--lr", type=float, default=5e-4, help="AlphaGo used 1e-3; the first run at 1e-3 collapsed")
    ap.add_argument("--kl-weight", type=float, default=0.1, help="anchor to the SL policy")
    ap.add_argument("--rollback-drop", type=float, default=0.15,
                    help="revert to the best checkpoint if win rate vs SL falls this far below the best")
    ap.add_argument("--sample-fraction", type=float, default=0.25, help="fraction of learner moves used per game")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--value-weight", type=float, default=0.5)
    ap.add_argument("--no-baseline", dest="baseline", action="store_false")
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--pool-every", type=int, default=200, help="add learner snapshot to opponent pool")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-games", type=int, default=128)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--referee", choices=["gnugo", "cleanup"], default="gnugo")
    ap.add_argument("--referee-procs", type=int, default=10)
    ap.add_argument("--referee-level", type=int, default=5)
    ap.add_argument("--no-pipeline", dest="pipeline", action="store_false",
                    help="judge each batch before playing the next (no one-update policy lag)")
    args = ap.parse_args()

    device = pick_device()
    run_dir = Path("checkpoints") / args.run
    pool_dir = run_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "metrics.jsonl"
    games_dir = run_dir / "games"
    games_dir.mkdir(exist_ok=True)
    cfg = SelfPlayConfig()
    referee = Referee(args.referee_procs, args.referee_level, cfg.komi) if args.referee == "gnugo" else None

    learner, sl_ck = load_model(args.sl, device)
    opponent, _ = load_model(args.sl, device)
    reference, _ = load_model(args.sl, device)  # fixed SL net for evaluation
    for net in (opponent, reference):
        net.eval()
    opt = torch.optim.SGD(learner.parameters(), lr=args.lr)

    it = 0
    pool = [str(Path(args.sl))]
    best = {"win_rate_vs_sl": 0.5, "iteration": 0}  # the SL start point itself scores 50%
    last = run_dir / "last.pt"
    best_path = run_dir / "best.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        learner.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        it, pool, best = ck["iteration"], ck["pool"], ck.get("best", best)
        if args.kl_ref is None:
            args.kl_ref = ck.get("kl_ref")  # keep the anchor chosen earlier when resuming
        print(f"resumed at iteration {it}, pool size {len(pool)}, best {best}")

    def save(path):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": learner.state_dict(), "optimizer": opt.state_dict(), "iteration": it, "pool": pool,
                    "best": best, "kl_ref": args.kl_ref, "config": sl_ck["config"], "args": vars(args)}, tmp)
        tmp.replace(path)

    if not best_path.exists():
        save(best_path)
    kl_net = reference
    if args.kl_ref and Path(args.kl_ref) != Path(args.sl):
        kl_net, _ = load_model(args.kl_ref, device)
        kl_net.eval()
    print(f"KL anchor: {args.kl_ref or args.sl}", flush=True)

    rng = np.random.default_rng(args.seed + it)
    judge_ex = ThreadPoolExecutor(1)
    pipelined = referee is not None and args.pipeline

    def judge_or_fallback(recs):
        failed = apply_referee(recs, referee, with_ownership=True)
        for i in failed:
            recs[i].judged = "fallback_tt"  # GNU Go failed: keep the raw Tromp-Taylor count
        return len(failed)

    def start_batch():
        """Play one batch on the GPU. When pipelined, GNU Go judges it on the CPU in the background
        while the GPU already plays the next batch (that batch then lags the policy by one update)."""
        t0 = time.time()
        opp_path = pool[rng.integers(len(pool))]
        opponent.load_state_dict(torch.load(opp_path, map_location=device, weights_only=False)["model"])
        learner.eval()
        if pipelined:
            recs = play_batch(learner, opponent, args.games, rng, device, cfg, defer_judging=True)
            fut = judge_ex.submit(judge_or_fallback, recs)
        else:
            recs = play_batch(learner, opponent, args.games, rng, device, cfg, referee=referee, with_ownership=True)
            fut = None
        return {"recs": recs, "fut": fut, "opp": opp_path, "t_play": time.time() - t0}

    pending = start_batch()
    t_last = time.time()
    while it < args.iterations:
        nxt = start_batch() if pipelined and it + 1 < args.iterations else None
        t_wait = time.time()
        if pending["fut"] is not None:
            pending["fut"].result()
        t_wait = time.time() - t_wait
        recs, opp_path, t_play = pending["recs"], pending["opp"], pending["t_play"]
        stats = reinforce_step(learner, kl_net, opt, recs, device, args, rng)
        it += 1
        t0 = t_last
        t_last = time.time()

        rec = {
            "iteration": it,
            "opponent": Path(opp_path).stem if Path(opp_path).parent.name == "pool" else "sl",
            "kl_ref": Path(args.kl_ref).stem if args.kl_ref else "sl",
            "win_rate": round(sum(r.learner_won for r in recs) / len(recs), 3),
            "avg_moves": round(float(np.mean([len(r.moves) for r in recs])), 1),
            "avg_cleanup": round(float(np.mean([len(r.cleanup) for r in recs])), 1),
            "cleanup_flip_rate": round(sum(r.flipped for r in recs) / len(recs), 3),
            "referee": f"gnugo {sum(r.judged == 'gnugo' for r in recs)}/{len(recs)}" if referee else "cleanup",
            "capped": sum(len(r.moves) >= cfg.max_moves for r in recs),
            **stats,
            "sec_play": round(t_play, 1),
            "sec_wait_judge": round(t_wait, 1),
            "sec_total": round(t_last - t0, 1),
        }
        if it % args.eval_every == 0 or it == args.iterations:
            wr = evaluate(learner, reference, device, args.eval_games, cfg, 10_000 + it, referee)
            rec["win_rate_vs_sl"] = round(wr, 3)
            if wr > best["win_rate_vs_sl"]:
                best = {"win_rate_vs_sl": wr, "iteration": it}
                save(best_path)
            elif wr < best["win_rate_vs_sl"] - args.rollback_drop:
                ck = torch.load(best_path, map_location=device, weights_only=False)
                learner.load_state_dict(ck["model"])
                lr = opt.param_groups[0]["lr"] * 0.5
                opt = torch.optim.SGD(learner.parameters(), lr=lr)
                rec["rollback"] = {"to_iteration": best["iteration"], "new_lr": lr}
                nxt = None  # that batch was played by the discarded weights
        save_games(games_dir / f"iter{it:05d}.npz", recs)
        print(json.dumps(rec), flush=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if it % args.pool_every == 0:
            snap = pool_dir / f"iter{it:05d}.pt"
            torch.save({"model": learner.state_dict(), "config": sl_ck["config"]}, snap)
            pool.append(str(snap))
        if it % args.save_every == 0 or it == args.iterations:
            save(last)
        if it < args.iterations:
            pending = nxt if nxt is not None else start_batch()
    judge_ex.shutdown()
    save(run_dir / "final.pt")
    if referee:
        referee.close()
    print("done")


if __name__ == "__main__":
    main()
