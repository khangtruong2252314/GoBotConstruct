import tensorflow as tf
from abc import abstractmethod
from src.util.board import Board
import json 
import numpy as np
import pyspiel


class Dataset:
    @staticmethod
    def tf_dataset():
        dataset = tf.data.Dataset.from_generator(Dataset,
                                            output_signature=(
                                                tf.TensorSpec(
                                                    shape=(None, 19, 19, 4),
                                                    dtype=tf.int32
                                                ),
                                                tf.TensorSpec(
                                                    shape=(None,),
                                                    dtype=tf.int32
                                                )
                                            ))
        return dataset 

    def __init__(self) -> None:
        dataset_dir = 'dataset/dataset.json'
        with open(dataset_dir) as f:
            lst = json.load(f)
        self.data = [[19 * u + v for u, v in small] for small in lst]
    
    def play(self, match_id: int):
        board = Board()
        result = [board.observation()]
        movement = self.data[match_id]
        for move in movement:
            board = board.try_step(move)
            result.append(board.observation())
        return result[:-1], movement 
    
    def __getitem__(self, index):
        result, movement = self.play(index)
        return np.array(result), np.array(movement)
    
    def __len__(self):
        return len(self.data) 
    
    def __iter__(self):
        for i in range(len(self)):
            try:
                yield self[i]
            except pyspiel.SpielError:
                pass 

    

class BreadthDataset:
    @staticmethod
    def tf_dataset():
        dataset = tf.data.Dataset.from_generator(BreadthDataset,
                                            output_signature=(
                                                tf.TensorSpec(
                                                    shape=(None, 19, 19, 4),
                                                    dtype=tf.int32
                                                ),
                                                tf.TensorSpec(
                                                    shape=(None,),
                                                    dtype=tf.int32
                                                )
                                            ))
        return dataset 

    def __init__(self) -> None:
        dataset_dir = 'dataset/dataset.json'
        with open(dataset_dir) as f:
            lst = json.load(f)
        self.data = [[19 * u + v for u, v in small] for small in lst]
    
    def play(self, match_id: int):
        board = Board()
        movement = self.data[match_id]
        result = [board.observation()]
        label = [1]
        for move in movement:
            wrong_move = [moving for moving in board.legal_actions() if moving != move]
            wrong_board = [board.try_step(action).observation() for action in wrong_move]
            result.extend(wrong_board)
            label.extend([0] * len(wrong_board))
        return result, label
    
    def __getitem__(self, index):
        result, movement = self.play(index)
        return np.array(result), np.array(movement)
    
    def __len__(self):
        return len(self.data) 
    
    def __iter__(self):
        for i in range(len(self)):
            try:
                yield self[i]
            except pyspiel.SpielError:
                pass 



