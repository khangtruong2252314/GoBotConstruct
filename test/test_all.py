import unittest 
from testwsl.src.util.board import *


class BoardTestCase(unittest.TestCase):
    def test_constructor(self):
        Board()

    def test_possible_moves(self):
        board = Board()
        board.legal_actions()

    def test_observation(self):
        board = Board()
        board.observation()

    def test_try_all(self):
        board = Board()
        board.try_all()
    
    def test_to_str(self):
        board = Board()
        board.to_string()
        str(board)

if __name__ == '__main__':
    unittest.main()