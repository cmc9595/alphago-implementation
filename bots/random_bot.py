import random
from go.board import Board, Move
from bots.base import Bot

class RandomBot(Bot):
    def select_move(self, board: Board) -> Move:
        moves = board.legal_moves()
        # resign 제외하고 싶으면 여기서 필터링
        moves = [m for m in moves if not m.is_resign]
        return random.choice(moves) if moves else Move.pass_move()
