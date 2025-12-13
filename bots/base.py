from abc import ABC, abstractmethod
from go.board import Board, Move

class Bot(ABC):
    @abstractmethod
    def select_move(self, board: Board) -> Move:
        ...
