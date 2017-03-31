#!/usr/bin/env python

import numpy as np
import time
board_size = 15
all_possible_winning_boards = []
for i in range(board_size):
    for j in range(board_size):
        new_board = np.zeros(board_size*board_size, dtype=int).reshape(board_size, board_size)
        # horizontal wins --
        if j <= board_size - 5:
            winning_board = new_board.copy()
            winning_board[i, j:j+5] = 1
            all_possible_winning_boards.append(winning_board)
        # vertical wins |
        if i <= board_size - 5:
            winning_board = new_board.copy()
            winning_board[i:i+5, j] = 1
            all_possible_winning_boards.append(winning_board)
        # left oblique wins /
        if i <= board_size - 5 and j >= 5:
            winning_board = new_board.copy()
            winning_board[ range(i,i+5), range(j,j-5,-1) ] = 1
            all_possible_winning_boards.append(winning_board)
        # right oblique wins \
        if i <= board_size - 5 and j <= board_size - 5:
            winning_board = new_board.copy()
            winning_board[ range(i,i+5), range(j,j+5) ] = 1
            all_possible_winning_boards.append(winning_board)
all_possible_winning_boards = np.array(all_possible_winning_boards)


opponent_board = (np.random.randint(10, size=board_size*board_size).reshape(board_size, board_size) > 5).astype(int)

print opponent_board

#print np.logical_and(all_possible_winning_boards[0],opponent_board).astype(int)
t0 = time.time()
for _ in range(10000):
    n = len(all_possible_winning_boards) - np.any(np.logical_and(all_possible_winning_boards,opponent_board[np.newaxis,:]), axis=(1,2)).sum()
t1 = time.time()
print(n)
print("-- %.2f s --" % (t1-t0))

def test(all_possible_winning_boards, opponent_board):
    opponent_board = np.ma.array(opponent_board)
    result = []
    for apwb in all_possible_winning_boards:
        opponent_board.mask = np.logical_not(apwb)
        result.append(np.any(opponent_board))
    n = sum(result)
    return n
for _ in range(100):
    n = test(all_possible_winning_boards, opponent_board)
t2 = time.time()
print(len(all_possible_winning_boards) - n)
print("-- %.2f s --" % (t2-t1))

    

