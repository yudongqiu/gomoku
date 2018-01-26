#!/usr/bin/env python

from __future__ import print_function, division
import itertools, time, copy
import collections, random
import os, pickle
import numba
import numpy as np
import tflearn

board_size = 15
estimate_level = 1
t_random = 0.01 # controls how random the bonus for level=0
show_q = False
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

    global board_size
    board, last_move, playing, board_size = state
    strategy.playing = playing

    other_player = int(not playing)
    my_stones = board[playing]
    opponent_stones = board[other_player]

    if playing == 0: # if i'm black
        strategy.zobrist_me = strategy.zobrist_black
        strategy.zobrist_opponent = strategy.zobrist_white
    else:
        strategy.zobrist_me = strategy.zobrist_white
        strategy.zobrist_opponent = strategy.zobrist_black

    last_move = (last_move[0]-1, last_move[1]-1)

    # build new state representation
    state = np.zeros(board_size**2, dtype=np.int8).reshape(board_size, board_size)
    strategy.zobrist_code = 0
    for i,j in my_stones:
        state[i-1,j-1] = 1
        strategy.zobrist_code ^= strategy.zobrist_me[i-1, j-1]
    for i,j in opponent_stones:
        state[i-1,j-1] = -1
        strategy.zobrist_code ^= strategy.zobrist_opponent[i-1, j-1]

    # clear the U cache
    U_stone.cache = dict()

    alpha = -2.0
    beta = 2.0
    empty_spots_left = np.sum(state==0)
    start_level = -1
    best_move, best_q = best_action_q(state, strategy.zobrist_code, empty_spots_left, last_move, alpha, beta, 1, start_level)

    state[best_move] = 1
    # update zobrist_code with my move
    strategy.zobrist_code ^= strategy.zobrist_me[best_move]
    # store the win rate of this move
    if strategy.zobrist_code not in strategy.learndata and best_q != None:
        strategy.learndata[strategy.zobrist_code] = [state, best_q, 1]

    game_finished = False
    new_u = 0
    if i_win(state, best_move, 1):
        new_u = 1.0
        game_finished = True
    elif i_lost(state, 1):
        new_u = -1.0
        game_finished = True
    elif empty_spots_left <= 2:
        new_u = 0.0
        game_finished = True

    if game_finished and strategy.started_from_beginning is True:
        #print("best_q = %f"%best_q)
        discount = 0.9
        #discount_factor = 0.9
        for prev_state_zobrist_code in strategy.hist_states[::-1]:
            st, u, n_visited = strategy.learndata[prev_state_zobrist_code]
            n_visited += 1
            new_u = u + discount * (new_u - u) / (n_visited**0.8) # this is the learning rate
            strategy.learndata[prev_state_zobrist_code] = (st, new_u, n_visited)
            print("Updated U of %d from %f to %f"%(prev_state_zobrist_code, u, new_u))
        print("Updated win rate of %d states" % len(strategy.hist_states))
        strategy.started_from_beginning = False # we only update once
    elif best_q != None:
        # record the history states
        strategy.hist_states.append(strategy.zobrist_code)
    if show_q and best_q != None:
        print("best_q = %f"%best_q)
    # return the best move
    return (best_move[0]+1, best_move[1]+1)



level_max_n = [10]*20
def best_action_q(state, zobrist_code, empty_spots_left, last_move, alpha, beta, player, level):
    "Return the optimal action for a state"
    if empty_spots_left == 0: # Board filled up, it's a tie
        return None, 0.0
    move_interest_values = best_action_q.move_interest_values
    move_interest_values.fill(0) # reuse the same array
    is_first_move = False
    if level == -1:
        level = 0
        is_first_move = True

    #verbose = is_first_move 
    verbose = False
    n_moves = level_max_n[level]
    interested_moves = tf_interest_moves(state, move_interest_values, player, n_moves, verbose)

    best_move = interested_moves[0] # continue to play even I'm losing

    if player == 1:
        max_q = -1.0
        max_bonused_q = 0.0
        for current_move in interested_moves:
            current_move = (current_move[0], current_move[1]) # convert into tuple
            q = Q_stone(state, zobrist_code, empty_spots_left, current_move, alpha, beta, player, level+1)
            if is_first_move:
                bonus_q = abs(np.random.normal(0, t_random)) #/ (226-empty_spots_left)**2
                if q + bonus_q > max_q:
                    max_q = q + bonus_q
                    best_move = current_move
                    max_bonused_q = bonus_q
            else:
                if q > alpha: alpha = q
                if q > max_q:
                    max_q = q
                    best_move = current_move
                if max_q >= 1.0 or beta <= alpha:
                    break
        best_q = max_q - max_bonused_q
    elif player == -1:
        min_q = 1.0
        for current_move in interested_moves:
            current_move = (current_move[0], current_move[1])
            q = Q_stone(state, zobrist_code, empty_spots_left, current_move, alpha, beta, player, level+1)
            if q < beta: beta = q
            if q < min_q:
                min_q = q
                best_move = current_move
            if q <= -1.0 or beta <= alpha:
                break
        best_q = min_q
    return best_move, best_q

def tf_interest_moves(state, move_interest_values, player, n_moves, verbose):
    empty_spots_r, empty_spots_c = np.nonzero(state==0)
    interest_moves = zip(empty_spots_r, empty_spots_c)
    n_all = len(empty_spots_r)
    if n_all < n_moves:
        n_moves = n_all
    black_or_white = 1 # default black
    if (player == 1 and strategy.playing == 1) or (player == -1 and strategy.playing == 0):
        black_or_white = 0
    all_tf_states = tf_predict_u.all_interest_states[:n_all]
    all_tf_states[:,:,:,0] = (state == player)
    all_tf_states[:,:,:,1] = (state == -player)
    all_tf_states[:,:,:,2] = black_or_white
    # measure the win rate of all possible moves
    for i,move in enumerate(interest_moves):
        r, c = move
        all_tf_states[i,r,c,0] = 1
    predict_u = np.array(tf_predict_u.model.predict(all_tf_states)).flatten()
    u_sort = np.argsort(predict_u)[::-1][:n_moves]
    result_moves = [interest_moves[i] for i in u_sort]
    if verbose == True:
        print("There are %d interested_moves" % n_moves)
        for i in xrange(n_moves):
            print('%d, %d  :  %f' % (result_moves[i][0], result_moves[i][1], predict_u[u_sort[i]]))
    return result_moves



def Q_stone(state, zobrist_code, empty_spots_left, current_move, alpha, beta, player, level):
    # update the state
    state[current_move] = player
    # update the zobrist code for the new state
    if player == 1:
        move_code = strategy.zobrist_me[current_move]
    else:
        move_code = strategy.zobrist_opponent[current_move]
    new_zobrist_code = zobrist_code ^ move_code

    result = U_stone(state, new_zobrist_code, empty_spots_left-1, current_move, alpha, beta, player, level)
    # revert the changes for the state
    state[current_move] = 0
    return result

def U_stone(state, zobrist_code, empty_spots_left, last_move, alpha, beta, player, level):
    try:
        return U_stone.cache[zobrist_code]
    except:
        pass
    if i_will_win(state, last_move, player):
        result = 1.0 if player == 1 else -1.0
    elif level >= estimate_level:
        try:
            if player == 1:
                return strategy.learndata[zobrist_code][1]
            else:
                return -strategy.opponent_learndata[zobrist_code][1]
        except KeyError:
            pass
        result = tf_predict_u(state, zobrist_code, empty_spots_left, last_move, player)
    else:
        best_move, best_q = best_action_q(state, zobrist_code, empty_spots_left, last_move, alpha, beta, -player, level)
        result = best_q
    U_stone.cache[zobrist_code] = result
    return result


def tf_predict_u(state, zobrist_code, empty_spots_left, last_move, player):
    "Generate the best moves, use the neural network to predict U, return the max U"
    if empty_spots_left == 0: # Board filled up, it's a tie
        return 0.0
    empty_spots_r, empty_spots_c = np.nonzero(state==0)
    interested_moves = zip(empty_spots_r, empty_spots_c)
    n_all = len(empty_spots_r)
    # calculate all possible moves of next player, pick the max and return negative as my win rate
    next_player = -player
    # find the known moves among interested_moves
    tf_moves = [] # all unknown moves will be evaluated by tf_evaluate_max_u
    max_q = -1.0
    zobrist_map = strategy.zobrist_me if next_player == 1 else strategy.zobrist_opponent
    learndata = strategy.learndata if next_player == 1 else strategy.opponent_learndata
    for this_move in interested_moves:
        this_move = (this_move[0], this_move[1])
        this_zobrist_code = zobrist_code ^ zobrist_map[this_move]
        try:
            max_q = max(max_q, learndata[this_zobrist_code][1])
        except KeyError:
            tf_moves.append(this_move)

    # run tensorflow to evaluate all unknown moves and find the largest q
    n_tf = len(tf_moves)
    if n_tf > 0:
        all_interest_states = tf_predict_u.all_interest_states[:n_tf] # we only need a slice of the big array
        if next_player == -1: # if next is opponent
            all_interest_states[:,:,:,0] = (state == -1)
            all_interest_states[:,:,:,1] = (state == 1)
            all_interest_states[:,:,:,2] = 1 if strategy.playing == 1 else 0 # if I'm white, next player is black
        elif next_player == 1: # if next is me
            all_interest_states[:,:,:,0] = (state == 1)
            all_interest_states[:,:,:,1] = (state == -1)
            all_interest_states[:,:,:,2] = 1 if strategy.playing == 0 else 0 # if I'm black, next is me so black
        for i,current_move in enumerate(tf_moves):
            ci, cj = current_move
            all_interest_states[i,ci,cj,0] = 1 # put current move down
        predict_y = tf_predict_u.model.predict(all_interest_states)
        predict_y = np.array(predict_y).flatten()
        tf_y = np.max(predict_y)
        max_q = max(max_q, tf_y)
    return max_q * next_player # if next_player is opponent, my win rate is negative of his

@numba.jit(nopython=True,nogil=True)
def i_win(state, last_move, player):
    """ Return true if I just got 5-in-a-row with last_move """
    r, c = last_move
    # try all 4 directions, the other 4 is included
    directions = [(1,1), (1,0), (0,1), (1,-1)]
    for dr, dc in directions:
        line_length = 1 # last_move
        # try to extend in the positive direction (max 4 times)
        ext_r = r
        ext_c = c
        for _ in range(5):
            ext_r += dr
            ext_c += dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length += 1
            else:
                break
        # try to extend in the opposite direction
        ext_r = r
        ext_c = c
        for _ in range(6-line_length):
            ext_r -= dr
            ext_c -= dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length += 1
            else:
                break
        if line_length is 5:
            return True # 5 in a row
    return False

@numba.jit(nopython=True,nogil=True)
def i_lost(state, player):
    for r in range(board_size):
        for c in range(board_size):
            if state[r,c] == 0 and i_win(state, (r,c), -player):
                return True
    return False

@numba.jit(nopython=True,nogil=True)
def i_will_win(state, last_move, player):
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
        line_length = 1 # last_move
        # try to extend in the positive direction (max 5 times to check overline)
        ext_r = r
        ext_c = c
        skipped_1 = 0
        for i in range(5):
            ext_r += dr
            ext_c += dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length += 1
            elif skipped_1 is 0 and state[ext_r, ext_c] == 0:
                skipped_1 = i+1 # allow one skip and record the position of the skip
            else:
                break
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
        for i in range(6-line_length_back):
            ext_r -= dr
            ext_c -= dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length_back += 1
            elif skipped_2 is 0 and state[ext_r, ext_c] == 0:
                skipped_2 = i + 1
            else:
                break

        if line_length_back == 6:
            # we found 6 stones in a row, this is overline, skip this entire line
            continue
        elif line_length_back == 5:
            if (skipped_2 == 0) or (skipped_2 == (6-line_length_no_skip)):
                # we found 5 stones in a row, because the backward counting is not skipped in the middle
                return True
                # else there is an empty spot in the middle of 6 stones, it's not a hard 4 any more
        elif line_length_back == 4:
            # here we have only 4 stones, if skipped in back count, it's a hard 4
            if skipped_2 is not 0:
                n_hard_4 += 1 # backward hard 4
                if n_hard_4 == 2:
                    return True # two hard 4
        # here we check if there's a hard 4 in the forward direction
        # extend the forward line to the furthest "unskipped" stone
        if skipped_2 == 0:
            line_length += line_length_back - line_length_no_skip
        else:
            line_length += skipped_2 - 1
        # hard 4 only if forward length is 4, if forward reaches 5 or more, it's going to be overline
        if line_length == 4 and skipped_1 is not 0:
            n_hard_4 += 1 # forward hard 4
            if n_hard_4 == 2:
                return True # two hard 4 or free 4
    return False

def initialize():
    # initialize zobrist for u caching
    if not hasattr(strategy, 'zobrist_me'):
        np.random.seed(20180104) # use the same random matrix for storing
        strategy.zobrist_black = np.random.randint(np.iinfo(np.int64).max, size=board_size**2).reshape(board_size,board_size)
        strategy.zobrist_white = np.random.randint(np.iinfo(np.int64).max, size=board_size**2).reshape(board_size,board_size)
        # reset the random seed to random for other functions
        np.random.seed()

    if not hasattr(best_action_q, 'move_interest_values'):
        best_action_q.move_interest_values = np.zeros(board_size**2, dtype=np.float32).reshape(board_size,board_size)

    if not hasattr(strategy, 'learndata'):
        strategy.learndata = dict()

    if not hasattr(tf_predict_u, 'all_interest_states'):
        tf_predict_u.all_interest_states = np.zeros(board_size**4 * 3, dtype=np.int8).reshape(board_size**2, board_size, board_size, 3)

    if not hasattr(tf_predict_u, 'cache'):
        tf_predict_u.cache = dict()

def reset():
    strategy.hist_states = []
    strategy.started_from_beginning = True

def board_show(stones):
    if isinstance(stones, np.ndarray):
        stones = {(s1,s2) for s1, s2 in stones}
    print(' '*4 + ' '.join([chr(97+i) for i in xrange(board_size)]))
    print (' '*3 + '='*(2*board_size))
    for x in xrange(1, board_size+1):
        row = ['%2s|'%x]
        for y in xrange(1, board_size+1):
            if (x-1,y-1) in stones:
                c = 'x'
            else:
                c = '-'
            row.append(c)
        print (' '.join(row))

def print_state(state):
    assert isinstance(state, np.ndarray)
    print(' '*4 + ' '.join([chr(97+i) for i in xrange(board_size)]))
    print (' '*3 + '='*(2*board_size))
    for x in xrange(1, board_size+1):
        row = ['%2s|'%x]
        for y in xrange(1, board_size+1):
            if state[x-1,y-1] == 1:
                c = 'o'
            elif state[x-1,y-1] == -1:
                c = 'x'
            else:
                c = '-'
            row.append(c)
        print (' '.join(row))

def test():
    state = np.zeros(board_size**2, dtype=np.int8).reshape(board_size, board_size)
    state[(8,8,8,8,8,8),(4,5,6,7,8,9)] = 1
    lastmove = (8,7)
    assert i_will_win(state, lastmove, 1) == False
    state[8,8] = 0
    assert i_will_win(state, lastmove, 1) == False
    state[8,9] = 0
    assert i_will_win(state, lastmove, 1) == True

    print("All test passed!")

if __name__ == '__main__':
    test()

def draw_state(state):
    board_size = 15
    print(' '*4 + ' '.join([chr(97+i) for i in xrange(board_size)]))
    print (' '*3 + '='*(2*board_size))
    me = state[0,0,2]
    for x in xrange(1, board_size+1):
        row = ['%2s|'%x]
        for y in xrange(1, board_size+1):
            if state[x-1,y-1,0] == 1:
                c = 'x' if me == 1 else 'o'
            elif state[x-1,y-1,1] == 1:
                c = 'o' if me == 1 else 'x'
            else:
                c = '-'
            row.append(c)
        print (' '.join(row))
