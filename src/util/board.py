from numpy._core.multiarray import array as array
import pyspiel 
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass


class IBoard(ABC):
    @abstractmethod
    def try_step(self, step: int):
        pass

    @abstractmethod
    def legal_actions(self) -> list[int]:
        pass

    @abstractmethod
    def observation(self) -> np.array:
        pass

    @abstractmethod
    def try_all(self) -> list:
        pass

    @abstractmethod
    def to_string(self) -> str:
        raise NotImplemented('to_string')
    
    @abstractmethod
    def try_all_tensor(self):
        pass

    @abstractmethod
    def end_game(self):
        pass

    @abstractmethod
    def winner(self):
        pass

    def __str__(self) -> str:
        return self.to_string()


@dataclass 
class GameState:
    game: pyspiel.Game
    current_state: pyspiel.State
    serialized: str 


class Board(IBoard):
    def __init__(self, load_state: None | GameState = None):
        if load_state is None:
            game = pyspiel.load_game('go')
            current_state = game.new_initial_state()
            serialized = current_state.serialize()
            self.game_state = GameState(game=game,
                                        current_state=current_state,
                                        serialized=serialized)
        else:
            self.game_state = load_state

    def try_step(self, step: int) -> IBoard:
        copy_serialize = self.game_state.current_state.serialize()
        copy_state = self.game_state.game.deserialize_state(copy_serialize)
        copy_state.apply_action(step)

        new_game_state = GameState(self.game_state.game,
                                   copy_state,
                                   copy_state.serialize())
        return Board(new_game_state)
    
    def legal_actions(self) -> list[int]:
        return self.game_state.current_state.legal_actions()
    
    def observation(self) -> np.array:
        tensor_shape = self.game_state.game.observation_tensor_shape()
        observation = self.game_state.current_state.observation_tensor(1)
        reshaped = np.reshape(observation, tensor_shape)
        reshaped = np.transpose(reshaped, (1, 2, 0))
        return reshaped 
    
    def try_all(self) -> list[IBoard]:
        all_legal_moves = self.legal_actions()
        all_possible_boards = [self.try_step(x) for x in all_legal_moves]
        return all_possible_boards
    
    def try_all_tensor(self) -> np.array:
        arrays = [x.observation() for x in self.try_all()]
        return np.array(arrays)
    
    def to_string(self) -> str:
        return str(self.game_state.current_state)
    
    def end_game(self) -> bool:
        return self.game_state.current_state.is_terminal()
    
    def winner(self) -> int:
        arr = self.game_state.current_state.rewards()
        return 0 if arr[0] == 1.0 else 1
    

    


        