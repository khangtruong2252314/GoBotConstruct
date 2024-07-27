from src.util.agent import Samurai
from src.util.board import Board 
from src.util.katana import Katana, Core_v2, Core
from src.util.dataset import BreadthDataset
import random 
from src.util.config import *
from abc import ABC, abstractmethod
from os import system
import time
import tf_keras as keras


class TrainEnvironment(ABC):
    @abstractmethod
    def evolution(self):
        pass 


class Daimyo(TrainEnvironment):
    def __init__(self):
        super().__init__()
        self.generation_count = GEN_COUNT
        self.generation = [Katana() for _ in range(self.generation_count)]

    def display(self):
        print('Training')
        current_samurai = Samurai(Katana())
        next_samurai = Samurai(Katana())
        battle = Board()
        while not battle.end_game():
            battle = current_samurai.attack(battle)
            system('clear')
            current_samurai, next_samurai = next_samurai, current_samurai
            print(battle)
    
    def evolution(self):
        north, south = self.generation[: self.generation_count // 2], self.generation[ - self.generation_count // 2:]

        def survive(k1: Katana, k2: Katana):
            print('Training')
            start = time.time()
            current_samurai = Samurai(k1)
            next_samurai = Samurai(k2)
            battle = Board()
            while not battle.end_game():
                battle = current_samurai.attack(battle)
                current_samurai, next_samurai = next_samurai, current_samurai
            print('Finished')
            print(f'Single thread executed: {time.time() - start}')
            return k1 if battle.winner() == 0 else k2

        survivor = [survive(u, v) for u, v in zip(north, south)]
        mating_id = [(random.randint(0, self.generation_count // 2 - 1),
                        random.randint(0, self.generation_count // 2 - 1))]
        offspring = [survivor[i] * survivor[j] for i, j in mating_id]
        self.generation = survivor + offspring


class Shogun(TrainEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.core = Core()
        self.dataset = BreadthDataset.tf_dataset()
        self.epoch = EPOCH 
        inputs = keras.Input(shape=(19, 19, 4))
        self.model = keras.Model(inputs, self.core(inputs))
        self.model.compile(loss=keras.losses.binary_crossentropy)

    def evolution(self):
        self.model.fit(self.dataset, 
                       epochs=self.epoch) 
            


