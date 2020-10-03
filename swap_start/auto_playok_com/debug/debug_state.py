# coding: utf-8
import numpy as np
import AI_debug
from tf_model import load_existing_model

board_size=15
def read_board_state(f):
    # default
    black_stones = set()
    white_stones = set()
    board = [black_stones, white_stones]
    last_move = None
    playing = 0
    # read and parse board
    for line in open(f):
        if '|' in line:
            line_idx, contents = line.split('|', maxsplit=1)
            row_i = int(line_idx) - 1
            ls = contents.split()
            if len(ls) == board_size:
                for col_j, s in enumerate(ls):
                    stone = (row_i+1, col_j+1)
                    if s == 'x':
                        black_stones.add(stone)
                    elif s == 'X':
                        black_stones.add(stone)
                        last_move = stone
                        playing = 1
                    elif s == 'o':
                        white_stones.add(stone)
                    elif s == 'O':
                        white_stones.add(stone)
                        last_move = stone
                        playing = 0
                    elif s == '-':
                        pass
                    else:
                        print(f'found unknown stone: {s}')
    board_state = [board, last_move, playing, board_size]
    return board_state

board_state = read_board_state('debug_board.txt')


model = load_existing_model('tf_model.h5')
AI_debug.tf_predict_u.model = model
AI_debug.initialize()    

print(AI_debug.strategy(board_state))
