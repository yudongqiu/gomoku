#!/usr/bin/env python

from __future__ import print_function, division
import itertools, time, copy
import collections, random
import os, pickle
import numba
import numpy as np

board_size = 15
def strategy(state):
    """ AI's strategy """

    """ Information provided to you:

    state = (board, last_move, playing, board_size)
    board = (x_stones, o_stones)
    stones is a set contains positions of one player's stones. e.g.
        x_stones = {(8,8), (8,9), (8,10), (8,11)}
    playing = 0|1, the current player's index

    Your strategy will return a position code for the next stone, e.g. (8,7)
    """
    # load information
    global board_size
    board, last_move, playing, board_size = state
    # generate necessary information
    initialize()
    # prepare internal state of the game
    other_player = int(not playing)
    my_stones = { (i-1,j-1) for i,j in board[playing] }
    opponent_stones = { (i-1,j-1) for i,j in board[other_player] }
    state = (my_stones, opponent_stones)
    # put the first stone in the center if it's the start of the game
    center = int((board_size+1)/2)
    if last_move is None:
        return (center, center)
    elif len(my_stones) == 0:
        # Assuming the first stone was put on the center
        return random.choice([(center-1, center+1), (center-1,center)])
    U_stone.cache = dict()
    alpha = -1.0
    beta = 2.0
    best_move, best_q = best_action_q(state, last_move, alpha, beta, True, 0)
    if best_q == 0:
        return (0,0) # admit defeat if I'm losing
    else:
        i, j = best_move
        return (i+1, j+1)
    #return best_move

def best_action_q(state, last_move, alpha, beta, maximizing_player, level):
    "Return the optimal action for a state"
    best_move = (0,0) # admit defeat if all moves have 0 win rate
    possible_moves = available_positions(state, 2)
    if len(possible_moves) == 1:
        current_move = possible_moves.pop()
        return current_move, 0.5# q
    n_candidates = len(possible_moves)
    if maximizing_player:
        max_q = 0.0
        for current_move in possible_moves:
            q = Q_stone(state, current_move, alpha, beta, maximizing_player, level+1)
            if q > alpha: alpha = q
            if q > max_q:
                if level == 0:
                    print(current_move, q)
                max_q = q
                best_move = current_move
            if q == 1.0 or beta <= alpha:
                break
        best_q = max_q
    else:
        min_q = 1.0
        for current_move in possible_moves:
            q = Q_stone(state, current_move, alpha, beta, maximizing_player, level+1)
            if q < beta: beta = q
            if q < min_q:
                min_q = q
                best_move = current_move
            if q == 0.0 or beta <= alpha:
                break
        best_q = min_q
    return best_move, best_q

@numba.jit(nopython=True, nogil=True)
def available_positions(state, dist=1):
    my_stones, opponent_stones = state
    positions = set()
    limiting = False # limiting positions if someone is winning
    for x in range(0, board_size+0):
        for y in range(0, board_size+0):
            stone = (x,y)
            if stone not in my_stones and stone not in opponent_stones:
                if limiting is False:
                    if i_win(my_stones, stone) or i_win(opponent_stones,stone):
                        limiting = True
                        positions = {stone}
                    elif near_any_stone(stone, state, dist):
                        positions.add(stone)
                else:
                    if i_win(my_stones, stone) or i_win(opponent_stones,stone):
                        positions.add(stone)
    return positions

@numba.jit(nopython=True, nogil=True)
def near_any_stone(last_move, state, dist):
    r1, c1 = last_move
    for r2, c2 in state[0]:
        if abs(r2-r1) <= dist and abs(c2-c1) <= dist:
            return True
    for r2, c2 in state[1]:
        if abs(r2-r1) <= dist and abs(c2-c1) <= dist:
            return True
    return False

def Q_stone(state, current_move, alpha, beta, maximizing_player, level):
    my_stones, opponent_stones = state
    if maximizing_player:
        new_my_stones = my_stones.copy()
        new_my_stones.add(current_move)
        my_stones = new_my_stones
    else:
        new_opponent_stones = opponent_stones.copy()
        new_opponent_stones.add(current_move)
        opponent_stones = new_opponent_stones
    state = (my_stones, opponent_stones)
    return U_stone(state, current_move, alpha, beta, maximizing_player, level)

def U_stone(state, last_move, alpha, beta, maximizing_player, level):
    my_stones, opponent_stones = state
    if maximizing_player is True: # put the stones of the current player first
        key = (frozenset(my_stones), frozenset(opponent_stones))
    else:
        key = (frozenset(opponent_stones), frozenset(my_stones))
    try:
        cached_result = U_stone.cache[key] # the first player's win rate
        if maximizing_player is True:
            return cached_result
        else:
            return 1.0 - cached_result
    except:
        pass

    MC_start_level = 3
    if maximizing_player is True and i_will_win(my_stones, opponent_stones, last_move):
        return 1.0
    elif maximizing_player is False and i_will_win(opponent_stones, my_stones, last_move):
        return 0.0
    elif level >= MC_start_level:
        result = estimate_U(state)
    else:
        best_move, best_q = best_action_q(state, last_move, alpha, beta, not maximizing_player, level)
        result = best_q

    if maximizing_player:
        cached_result = result
    else:
        cached_result = 1.0 - result

    U_stone.cache[key] = cached_result

    return result

def estimate_U(state):
    my_stones, opponent_stones = state
    apwb = estimate_U.all_possible_winning_boards
    board = estimate_U.board # matrix representation of the board
    # possible wins for me
    board.fill(0)
    board[zip(*opponent_stones)] = 1
    my_possible_wins = len(apwb) - np.any(np.logical_and(apwb,board[np.newaxis,:]), axis=(1,2)).sum()
    # possible wins for opponent
    board.fill(0)
    board[zip(*my_stones)] = 1
    opponent_possible_wins = len(apwb) - np.any(np.logical_and(apwb,board[np.newaxis,:]), axis=(1,2)).sum()
    return my_possible_wins / (my_possible_wins + opponent_possible_wins)

def prepare_all_possible_winning_boards(board_size):
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
    return np.array(all_possible_winning_boards)

@numba.jit(nopython=True,nogil=True)
def i_win(my_stones, last_move):
    """ Return true if I just got 5-in-a-row with last_move """
    if len(my_stones) < 4: return False
    r, c = last_move
    # try all 4 directions, the other 4 is included
    directions = [(1,1), (1,0), (0,1), (1,-1)]
    for dr, dc in directions:
        line_length = 1 # last_move
        # try to extend in the positive direction (max 4 times)
        ext_r = r
        ext_c = c
        for i in range(4):
            ext_r += dr
            ext_c += dc
            if (ext_r, ext_c) in my_stones:
                line_length += 1
            else:
                break
        if line_length is 5:
            return True # 5 in a row
        # try to extend in the opposite direction
        ext_r = r
        ext_c = c
        for i in range(5-line_length):
            ext_r -= dr
            ext_c -= dc
            if (ext_r, ext_c) in my_stones:
                line_length += 1
            else:
                break
        if line_length is 5:
            return True # 5 in a row
    return False

@numba.jit(nopython=True,nogil=True)
def i_will_win(my_stones, blocking_stones, last_move):
    """ Return true if I will win next step if the opponent don't have 4-in-a-row.
    Winning Conditions:
        1. 5 in a row.
        2. 4 in a row with both end open. (free 4)
        3. 4 in a row with one missing stone x 2 (hard 4 x 2)
     """
    r, c = last_move
    # try all 4 directions, the other 4 is equivalent
    directions = [(1,1), (1,0), (0,1), (1,-1)]
    n_hard_4 = 0 # number of hard 4s found
    for dr, dc in directions:
        #print(dr, dc)
        line_length = 1 # last_move
        # try to extend in the positive direction (max 4 times)
        ext_r = r
        ext_c = c
        skipped_1 = 0
        for i in range(4):
            ext_r += dr
            ext_c += dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size: break
            if (ext_r, ext_c) in my_stones:
                line_length += 1
            elif skipped_1 is 0 and (ext_r, ext_c) not in blocking_stones:
                skipped_1 = i+1 # allow one skip and record the position of the skip
            else:
                break
        if line_length is 5:
            return True # 5 in a row
        #print("Forward line_length",line_length)
        # try to extend in the opposite direction
        ext_r = r
        ext_c = c
        skipped_2 = 0
        # the backward counting starts at the furthest "unskipped" stone
        if skipped_1 is not 0:
            line_length_back = skipped_1
        else:
            line_length_back = line_length
        line_length_no_skip = line_length_back
        for i in range(5-line_length_back):
            ext_r -= dr
            ext_c -= dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size: break
            if (ext_r, ext_c) in my_stones:
                line_length_back += 1
            elif skipped_2 is 0 and (ext_r, ext_c) not in blocking_stones:
                skipped_2 = i + 1
            else:
                break
        #print("Backward line_length",line_length_back)
        if line_length_back is 5:
            return True # 5 in a row
        if line_length_back == 4 and skipped_2 is not 0:
            n_hard_4 += 1 # backward hard 4
            if n_hard_4 == 2:
                return True # two hard 4

        #print("back n_hard_4 = ", n_hard_4)

        # extend the forward line to the furthest "unskipped" stone
        #print("line_length_back", line_length_back)
        if skipped_2 is 0:
            line_length += line_length_back - line_length_no_skip
        else:
            line_length += skipped_2 - 1
        if line_length >= 4 and skipped_1 is not 0:
            n_hard_4 += 1 # forward hard 4
            if n_hard_4 == 2:
                return True # two hard 4 or free 4
        #print('total n_hard_4', n_hard_4)
    return False

def initialize():
    if not hasattr(estimate_U, 'all_possible_winning_boards'):
        estimate_U.all_possible_winning_boards = prepare_all_possible_winning_boards(board_size)
        estimate_U.board = np.zeros(board_size*board_size, dtype=int).reshape(board_size, board_size)

def finish():
    return

def check():
    global board_size
    board_size = 15
    # check if i_win() is working properly
    assert i_win({(8,9), (8,11), (8,8), (8,10), (8,12)}, (8,10)) == True
    assert i_win({(8,9), (8,11), (8,8), (8,10)}, (8,12)) == True
    assert i_win({(8,10), (9,11), (8,8), (9,12), (7,9), (10,9), (11,12), (11,13)}, (10,12)) == True
    assert i_win({(8,10), (8,12), (8,8), (9,12), (7,9), (10,9), (11,12), (11,13)}, (10,12)) == False
    # check if i_will_win() is working properly
    # o - x x X x - o
    assert i_will_win({(8,9), (8,11), (8,8)}, {(8,6), (8,13)}, (8,10)) == True
    # o - x x X x o
    assert i_will_win({(8,9), (8,11), (8,8)}, {(8,6), (8,12)}, (8,10)) == False
    # o - x x X o
    #         x
    #
    #         x
    #         x
    assert i_will_win({(8,9), (8,8), (9,10), (11,10), (12,10)}, {(8,6), (8,11)}, (8,10)) == False
    # o - x x X x o
    #         x
    #
    #         x
    #         x
    assert i_will_win({(8,9), (8,8), (9,10), (11,10), (12,10)}, {(8,6), (8,11)}, (8,10)) == False
    # o - x x X x o
    #       x
    #
    #   x
    # x
    assert i_will_win({ (8,8), (8,9), (8,11), (9,9), (11,7), (12,6)}, {(8,6), (8,12)}, (8,10)) == True
    # | x x x X - x x x - - o
    assert i_will_win({(8,1), (8,2), (8,3), (8,6), (8,7), (8,8)}, {(8,11)}, (8,4)) == False
    # | x x - x X x x o
    assert i_will_win({(8,1), (8,2), (8,4), (8,6), (8,7)}, {(8,8)}, (8,5)) == False
    # | x x - x X - x x o
    assert i_will_win({(8,1), (8,2), (8,4), (8,7), (8,8)}, {(8,9)}, (8,5)) == True
    # | x x x - X - x x x o
    assert i_will_win({(8,1), (8,2), (8,3), (8,7), (8,8), (8,9)}, {(8,10)}, (8,5)) == True
    # | x - x X x - x o
    assert i_will_win({(8,1), (8,3), (8,5), (8,7)}, {(8,8)}, (8,4)) == True

    assert i_will_win({(8,8), (8,10), (9,9), (11,7), (11,9)}, {(7,7), (7,9), (8,7), (10,8), (11,8)}, (8,9)) == False
    print("All check passed!")

if __name__ == '__main__':
    import time
    check()
