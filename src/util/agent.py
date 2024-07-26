from src.util.katana import Katana 
from src.util.board import IBoard
from abc import ABC
import numpy as np


class Agent(ABC):
    pass


class Samurai(Agent):
    def __init__(self, katana: Katana):
        super().__init__()
        self.katana = katana 
    
    def attack(self, board: IBoard):
        possible_strategies = board.legal_actions()
        lst_board = board.try_all_tensor()
        scoring: np.ndarray = self.katana(lst_board)
        best = np.argmax(scoring)
        answer = possible_strategies[best]
        return board.try_step(answer)

    



