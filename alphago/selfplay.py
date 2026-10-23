"""Batched policy-vs-policy games (AlphaGo RL stage): many boards advance one move per step.

Game end follows the project's option C: the normal phase ends at two consecutive passes
(or a move cap). The winner is then decided by a referee. Preferred: GNU Go
(alphago.gnugo.Referee), which reads each group's life and death and also yields per-point
final ownership. Fallback when no referee is given: a cleanup phase resolves the board before
Tromp-Taylor scoring:
passing is forbidden while a cleanup move exists, and cleanup moves are restricted to empty
regions bordered by both colours (dame, areas holding dead stones) and must not be
self-atari. Filling one's own territory is never needed under area scoring and, when forced,
can kill living groups or break seki. Only normal-phase moves are used as training data;
cleanup only decides the winner.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from alphago import Board, Color, features_batch
from alphago.gnugo import ownership

SELF_ATARI = slice(23, 26)  # feature planes: move leaves own chain with one liberty


@dataclass
class SelfPlayConfig:
    board_size: int = 19
    komi: float = 7.5
    max_moves: int = 400  # normal-phase cap (superko is not enforced, so loops are cut here)
    cleanup_max: int = 200
    temperature: float = 1.0
    threads: int = 0


@dataclass
class GameRecord:
    moves: list[int] = field(default_factory=list)  # normal phase only, Black first
    cleanup: list[int] = field(default_factory=list)
    learner_black: bool = True
    score_before: float = 0.0  # Tromp-Taylor right after the normal phase (black - white - komi)
    score_after: float = 0.0  # official result (referee, or Tromp-Taylor after cleanup)
    dead: list[int] = field(default_factory=list)  # dead stones per the referee
    ownership: np.ndarray | None = None  # +1 Black / -1 White / 0 neutral per point (referee only)
    judged: str = "cleanup"

    @property
    def learner_won(self) -> bool:
        return (self.score_after > 0) == self.learner_black

    @property
    def flipped(self) -> bool:
        return (self.score_before > 0) != (self.score_after > 0)


@torch.no_grad()
def policy_logits(net, boards, device, threads=0) -> np.ndarray:
    x = torch.from_numpy(features_batch(boards, threads)).to(device).float()
    logits, _ = net(x)
    return logits.float().cpu().numpy()


def cleanup_mask(b: Board) -> np.ndarray:
    """Moves allowed during cleanup (pass only if none)."""
    nn_ = b.size * b.size
    mask = b.sensible_mask().astype(bool)
    contested = b.region_owner() == 3
    self_atari = b.features()[SELF_ATARI].any(0).ravel()
    mask[:nn_] &= contested & ~self_atari
    mask[nn_] = not mask[:nn_].any()
    return mask


def apply_referee(recs, referee, with_ownership=False, n=19, komi=7.5):
    """Judge games whose normal phase is over. Returns the indices the referee could not judge."""
    verdicts = referee.judge_many([r.moves for r in recs])
    failed = []
    for i, (r, v) in enumerate(zip(recs, verdicts)):
        if v is None:
            failed.append(i)
            continue
        r.score_after, r.dead = v
        r.judged = "gnugo"
        if with_ownership:
            r.ownership = ownership(r.moves, r.dead, n, komi)
    return failed


def play_batch(learner, opponent, n_games, rng: np.random.Generator, device, cfg: SelfPlayConfig,
               referee=None, with_ownership=False, defer_judging=False):
    """Play n_games between two nets (learner is Black in even-numbered games).

    With a referee, games stop after the normal phase and GNU Go decides the result; games it
    fails to judge fall back to the cleanup phase.

    defer_judging: stop every game after its normal phase and return unjudged records
    (judged == "pending"), so the caller can run apply_referee on CPU while the GPU already plays
    the next batch.
    """
    n = cfg.board_size
    pass_a = n * n
    boards = [Board(n, cfg.komi) for _ in range(n_games)]
    recs = [GameRecord(learner_black=(i % 2 == 0)) for i in range(n_games)]
    phase = ["normal"] * n_games
    cleanup_passes = [0] * n_games

    while True:
        active = [i for i in range(n_games) if phase[i] not in ("done", "judge")]
        if not active:
            pending = [i for i in range(n_games) if phase[i] == "judge"]
            if not pending:
                break
            failed = set(apply_referee([recs[i] for i in pending], referee, with_ownership, n, cfg.komi))
            for k, i in enumerate(pending):
                # Referee failed: resolve this game by the cleanup phase instead.
                phase[i] = "cleanup" if k in failed else "done"
            continue
        by_net = {0: [], 1: []}  # 0 = learner to move, 1 = opponent to move
        for i in active:
            black = boards[i].to_move == Color.BLACK
            by_net[0 if black == recs[i].learner_black else 1].append(i)
        for which, idxs in by_net.items():
            if not idxs:
                continue
            net = learner if which == 0 else opponent
            logits = policy_logits(net, [boards[i] for i in idxs], device, cfg.threads)
            # Gumbel-max: argmax(logits/T + Gumbel noise) samples from softmax(logits/T).
            noise = rng.gumbel(size=logits.shape) if cfg.temperature > 0 else 0.0
            for row, i in enumerate(idxs):
                b = boards[i]
                if phase[i] == "cleanup":
                    mask = cleanup_mask(b)
                    score = np.where(mask, logits[row], -np.inf)  # greedy while cleaning up
                else:
                    mask = b.sensible_mask().astype(bool)
                    t = max(cfg.temperature, 1e-6)
                    score = np.where(mask, logits[row] / t + (noise[row] if cfg.temperature > 0 else 0), -np.inf)
                a = int(np.argmax(score))
                b.play(a)
                rec = recs[i]
                if phase[i] == "normal":
                    rec.moves.append(a)
                    if b.game_over or len(rec.moves) >= cfg.max_moves:
                        rec.score_before = b.score()
                        if defer_judging:
                            rec.judged, rec.score_after = "pending", rec.score_before
                            phase[i] = "done"
                        else:
                            phase[i] = "judge" if referee is not None else "cleanup"
                else:
                    rec.cleanup.append(a)
                    cleanup_passes[i] = cleanup_passes[i] + 1 if a == pass_a else 0
                    if cleanup_passes[i] >= 2 or len(rec.cleanup) >= cfg.cleanup_max:
                        rec.score_after = b.score()
                        phase[i] = "done"
    return recs
