#!/usr/bin/env python

from __future__ import print_function
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

    global board_size
    board, last_move, playing, board_size = state

    initialize()

    other_player = int(not playing)
    my_stones = board[playing]
    opponent_stones = board[other_player]


    # put the first stone in the center if it's the start of the game
    center = int((board_size-1)/2)



    if last_move is None: # if it's the first move of the game
        best_move = (center, center)
        strategy.zobrist_code = strategy.zobrist_me[best_move]
        return (best_move[0]+1, best_move[1]+1)
    else:
        last_move = (last_move[0]-1, last_move[1]-1)
        # update zobrist_code with opponent last move
        strategy.zobrist_code ^= strategy.zobrist_opponent[last_move]

    if len(my_stones) == 0:
        # Assuming the first stone was put on the center
        best_move = random.choice([(center-1, center+1), (center-1,center)])
    else:
        # build new state representation
        state = np.zeros(board_size**2, dtype=np.int32).reshape(board_size, board_size)
        for i,j in my_stones:
            state[i-1,j-1] = 1
        for i,j in opponent_stones:
            state[i-1,j-1] = -1

        if strategy.zobrist_code in U_stone.cache:
            print("Calculated Move: %.3f" %U_stone.cache[strategy.zobrist_code])
        else:
            print("Didn't know this move!")

        # clear the U cache
        U_stone.cache = dict()

        alpha = -1.0
        beta = 2.0
        possible_moves = available_positions(state, 2)
        global move_interest
        best_move, best_q = best_action_q(state, strategy.zobrist_code, possible_moves, last_move, move_interest, alpha, beta, 1, 0)

        move_interest *= 0.9

    # update zobrist_code with my move
    strategy.zobrist_code ^= strategy.zobrist_me[best_move]
    # return the best move
    return (best_move[0]+1, best_move[1]+1)


move_interest = np.zeros(board_size**2, dtype=np.float32).reshape(board_size, board_size)

level_max_n = [30, 30, 10, 10, 8, 7, 7, 6, 5, 4, 4, 4, 4, 4, 4]
#level_max_n = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
def best_action_q(state, zobrist_code, possible_moves, last_move, move_interest, alpha, beta, player, level):
    "Return the optimal action for a state"

    best_move = (-1,-1) # admit defeat if all moves have 0 win rate
    my_possible_moves = possible_moves.copy()
    filter_moves(my_possible_moves, state, player)

    if len(my_possible_moves) == 1:
        current_move = my_possible_moves.pop()
        q = Q_stone(state, zobrist_code, possible_moves, current_move, move_interest, alpha, beta, player, level)
        return current_move, q

    # get the local interest
    local_interest = move_interest.copy() * 0.9 # current move_interest to be passed and modified by next level
    # increase the interest of moves that opponent will win
    adjust_threat_interests(my_possible_moves, state, local_interest, player)
    n_candidates = min(len(my_possible_moves), level_max_n[level])
    if player == 1:
        max_q = 0.0
        for _ in xrange(n_candidates):
            my_possible_moves = sorted(my_possible_moves, key=lambda m: local_interest[m])
            current_move = my_possible_moves.pop()
            q = Q_stone(state, zobrist_code, possible_moves, current_move, local_interest, alpha, beta, player, level+1)
            if q > move_interest[current_move]: # update the parent's interest matrix
                move_interest[current_move] = q
            if q > alpha: alpha = q
            if q > max_q:
                if level == 0:
                    print(current_move, q, "interest", local_interest[current_move])
                max_q = q
                best_move = current_move
            if q == 1.0 or beta <= alpha:
                break
        best_q = max_q
    elif player == -1:
        min_q = 1.0
        for _ in xrange(n_candidates):
            my_possible_moves = sorted(my_possible_moves, key=lambda m: local_interest[m])
            current_move = my_possible_moves.pop()
            q = Q_stone(state, zobrist_code, possible_moves, current_move, local_interest, alpha, beta, player, level+1)
            if (1 - q) > move_interest[current_move]:
                move_interest[current_move] = (1 - q)
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
    positions = set()
    for x in range(0, board_size+0):
        for y in range(0, board_size+0):
            stone = (x,y)
            if state[stone] == 0:
                xmin = max(x-dist, 0)
                xmax = min(x+dist+1, board_size)
                ymin = max(y-dist, 0)
                ymax = min(y+dist+1, board_size)
                for nx in range(xmin, xmax):
                    for ny in range(ymin, ymax):
                        if state[nx, ny] != 0:
                            positions.add(stone)
                            break
    return positions

@numba.jit(nopython=True, nogil=True)
def filter_moves(possible_moves, state, player):
    """ Filter the possible moves based on if one is winning or will win"""
    limited_moves = set()
    winning_moves = set()
    limited = False
    for move in possible_moves:
        # if the player is winning with this move
        if i_win(state, move, player):
            possible_moves.clear()
            possible_moves.add(move)
            return
        # if the other player is winning with this move
        elif i_win(state, move, -player):
            limited_moves.add(move)
            limited = True
        # if the other player is winning with this move
        elif limited is False and i_will_win(state, move, player):
            winning_moves.add(move)
    if limited is True:
        possible_moves.clear()
        possible_moves.update(limited_moves)
    elif len(winning_moves) > 0:
        possible_moves.clear()
        possible_moves.update(winning_moves)


@numba.jit(nopython=True, nogil=True)
def adjust_threat_interests(possible_moves, state, local_interest, player):
    directions = [(1,1), (1,0), (0,1), (1,-1)]
    for move in possible_moves:
        r, c = move
        interest_length = 0
        for dr, dc in directions:
            my_line_length = 1 # last_move
            opponent_line_length = 1
            # try to extend in the positive direction (max 4 times)
            ext_r = r
            ext_c = c
            skipped_1 = 0
            my_blocked = False
            opponent_blocked = False
            for i in range(4):
                ext_r += dr
                ext_c += dc
                if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                    break
                elif state[ext_r, ext_c] == player:
                    if my_blocked is True:
                        break
                    else:
                        my_line_length += 1
                        opponent_blocked = True
                elif state[ext_r, ext_c] == -player:
                    if opponent_blocked is True:
                        break
                    else:
                        opponent_line_length += 1
                        my_blocked = True
                elif skipped_1 is 0:
                    skipped_1 = i + 1 # allow one skip and record the position of the skip
                else:
                    break

            # the backward counting starts at the furthest "unskipped" stone
            forward_my_open = False
            forward_opponent_open = False
            if skipped_1 == 0:
                my_line_length_back = my_line_length
                opponent_line_length_back = opponent_line_length
            elif skipped_1 == 1:
                my_line_length_back = 1
                opponent_line_length_back = 1
                forward_my_open = True
                forward_opponent_open = True
            else:
                if my_blocked is False:
                    my_line_length_back = skipped_1
                    opponent_line_length_back = 1
                    forward_my_open = True
                else:
                    my_line_length_back = 1
                    opponent_line_length_back = skipped_1
                    forward_opponent_open = True


            my_line_length_no_skip = my_line_length_back
            opponent_line_length_no_skip = opponent_line_length_back

            ext_r = r
            ext_c = c
            skipped_2 = 0
            my_blocked = False
            opponent_blocked = False
            for i in range(4):
                ext_r -= dr
                ext_c -= dc
                if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                    break
                elif state[ext_r, ext_c] == player:
                    if my_blocked is True:
                        break
                    else:
                        my_line_length_back += 1
                        opponent_blocked = True
                elif state[ext_r, ext_c] == -player:
                    if opponent_blocked is True:
                        break
                    else:
                        opponent_line_length_back += 1
                        my_blocked = True
                elif skipped_2 is 0:
                    skipped_2 = i + 1
                else:
                    break


            backward_my_open = False
            backward_opponent_open = False
            if skipped_2 is 0:
                if my_blocked is False:
                    my_line_length += my_line_length_back - my_line_length_no_skip
                else:
                    opponent_line_length += opponent_line_length_back - opponent_line_length_no_skip
            elif skipped_2 == 1:
                backward_my_open = True
                backward_opponent_open = True
            else:
                if my_blocked is False:
                    my_line_length += skipped_2 - 1
                    backward_my_open = True
                else:
                    opponent_line_length += skipped_2 - 1
                    backward_opponent_open = True

            if forward_my_open is True and backward_my_open is True:
                interest_length += min(my_line_length, my_line_length_back)**3
            elif forward_opponent_open is True and backward_opponent_open is True:
                interest_length += min(opponent_line_length, opponent_line_length_back)**3
            elif forward_my_open is True and forward_opponent_open is True:
                line_length = max(my_line_length, opponent_line_length) - 1
                interest_length += (line_length) ** 3
            elif backward_my_open is True and backward_opponent_open is True:
                line_length_back = max(my_line_length_back, opponent_line_length_back) - 1
                interest_length += (line_length_back) ** 3
            elif forward_my_open is True:
                interest_length += (my_line_length-1) ** 3
            elif forward_opponent_open is True:
                interest_length += (opponent_line_length-1) ** 3
            elif backward_my_open is True:
                interest_length += (my_line_length_back-1) ** 3
            elif backward_opponent_open is True:
                interest_length += (opponent_line_length_back-1) ** 3
        move_interest = 1.0 - np.exp(-interest_length**2 * 0.002)
        if local_interest[move] < move_interest:
            local_interest[move] = move_interest


def Q_stone(state, zobrist_code, possible_moves, current_move, move_interest, alpha, beta, player, level):
    # update the board
    new_state = state.copy()
    new_state[current_move] = player
    # update the zobrist code for the new state
    if player == 1:
        move_code = strategy.zobrist_me[current_move]
    else:
        move_code = strategy.zobrist_opponent[current_move]
    new_zobrist_code = zobrist_code ^ move_code
    # update the possible_moves after current_move
    new_possible_moves = possible_moves.copy()
    update_possible_moves(new_possible_moves, new_state, current_move, 2)

    return U_stone(new_state, new_zobrist_code, new_possible_moves, current_move, move_interest, alpha, beta, player, level)

@numba.jit(nopython=True, nogil=True)
def update_possible_moves(possible_moves, state, current_move, dist):
    possible_moves.remove(current_move)
    x, y = current_move
    xmin = max(x-dist, 0)
    xmax = min(x+dist+1, board_size)
    ymin = max(y-dist, 0)
    ymax = min(y+dist+1, board_size)
    for nx in range(xmin, xmax):
        for ny in range(ymin, ymax):
            if state[nx, ny] == 0:
                possible_moves.add((nx,ny))

def U_stone(state, zobrist_code, possible_moves, last_move, move_interest, alpha, beta, player, level):
    try:
        return U_stone.cache[zobrist_code]
    except:
        pass

    estimate_level = 8
    if i_will_win(state, last_move, player):
        return 1.0 if player == 1 else 0.0
    elif level >= estimate_level:
        result = estimate_U(state, player)
    else:
        best_move, best_q = best_action_q(state, zobrist_code, possible_moves, last_move, move_interest, alpha, beta, -player, level)
        result = best_q

    U_stone.cache[zobrist_code] = result
    return result


@numba.jit(nopython=True, nogil=True)
def estimate_U(state, player):
    u = 0.0
    my_max_n = 0
    opponent_max_n = 0
    for i in range(board_size):
        for j in range(board_size):
            # horizontal wins --
            if j <= board_size - 5:
                my_blocked, opponent_blocked = False, False
                my_n, opponent_n = 0, 0
                for k in range(5):
                    if state[i, j+k] == -1:
                        my_blocked = True
                        opponent_n += 1
                    elif state[i, j+k] == 1:
                        opponent_blocked = True
                        my_n += 1
                    if my_blocked is True and opponent_blocked is True:
                        break
                if my_blocked is False:
                    u += 3 ** my_n
                    if my_n > my_max_n:
                        my_max_n = my_n
                if opponent_blocked is False:
                    u -= 3 ** opponent_n
                    if opponent_n > opponent_max_n:
                        opponent_max_n = opponent_n
            # vertical wins |
            if i <= board_size - 5:
                my_blocked, opponent_blocked = False, False
                my_n, opponent_n = 0, 0
                for k in range(5):
                    if state[i+k, j] == -1:
                        my_blocked = True
                        opponent_n += 1
                    elif state[i+k, j] == 1:
                        opponent_blocked = True
                        my_n += 1
                    if my_blocked is True and opponent_blocked is True:
                        break
                if my_blocked is False:
                    u += 3 ** my_n
                    if my_n > my_max_n:
                        my_max_n = my_n
                if opponent_blocked is False:
                    u -= 3 ** opponent_n
                    if opponent_n > opponent_max_n:
                        opponent_max_n = opponent_n
            # left oblique wins /
            if i <= board_size - 5 and j >= 4:
                my_blocked, opponent_blocked = False, False
                my_n, opponent_n = 0, 0
                for k in range(5):
                    if state[i+k, j-k] == -1:
                        my_blocked = True
                        opponent_n += 1
                    elif state[i+k, j-k] == 1:
                        opponent_blocked = True
                        my_n += 1
                    if my_blocked is True and opponent_blocked is True:
                        break
                if my_blocked is False:
                    u += 3 ** my_n
                    if my_n > my_max_n:
                        my_max_n = my_n
                if opponent_blocked is False:
                    u -= 3 ** opponent_n
                    if opponent_n > opponent_max_n:
                        opponent_max_n = opponent_n
            # right oblique wins \
            if i <= board_size - 5 and j <= board_size - 5:
                my_blocked, opponent_blocked = False, False
                my_n, opponent_n = 0, 0
                for k in range(5):
                    if state[i+k, j+k] == -1:
                        my_blocked = True
                        opponent_n += 1
                    elif state[i+k, j+k] == 1:
                        opponent_blocked = True
                        my_n += 1
                    if my_blocked is True and opponent_blocked is True:
                        break
                if my_blocked is False:
                    u += 3 ** my_n
                    if my_n > my_max_n:
                        my_max_n = my_n
                if opponent_blocked is False:
                    u -= 3 ** opponent_n
                    if opponent_n > opponent_max_n:
                        opponent_max_n = opponent_n
    if player == 1: # next move is opponent
        longer = 2 * (3 **opponent_max_n)  # one of the longest can get 1 longer
        block = 3 ** my_max_n
        u -= max(longer, block)
    else: # next move is me
        longer = 2 * (3 ** my_max_n)
        block = 3 ** opponent_max_n
        u += max(longer, block)

    if u > 0:
        result = 1.0 - 0.5 * np.exp(-u**2 * 0.0001)
    else:
        result = 0.5 * np.exp(-u**2 * 0.0001)
    return result


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
        for _ in range(4):
            ext_r += dr
            ext_c += dc
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length += 1
            else:
                break
        if line_length is 5:
            return True # 5 in a row
        # try to extend in the opposite direction
        ext_r = r
        ext_c = c
        for _ in range(5-line_length):
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
        #print(dr, dc)
        line_length = 1 # last_move
        # try to extend in the positive direction (max 4 times)
        ext_r = r
        ext_c = c
        skipped_1 = 0
        for i in range(4):
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
            if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                break
            elif state[ext_r, ext_c] == player:
                line_length_back += 1
            elif skipped_2 is 0 and state[ext_r, ext_c] == 0:
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
    # initialize zobrist for u caching
    if not hasattr(strategy, 'zobrist_me'):
        strategy.zobrist_me = np.random.randint(np.iinfo(np.int64).max, size=board_size**2).reshape(board_size,board_size)
    #if not hasattr(strategy, 'zobrist_opponent'):
        strategy.zobrist_opponent = np.random.randint(np.iinfo(np.int64).max, size=board_size**2).reshape(board_size,board_size)
    #if not hasattr(strategy, 'zobrist_code'):
        strategy.zobrist_code = 0
    if not hasattr(U_stone, 'cache'):
        U_stone.cache = dict()

def finish():
    return
    if not hasattr(U_stone, 'cachehigh'):
        return
    t0 = time.time()
    global n_cache
    if os.path.exists("cachehigh"): # if it has been updated by another AI
        prev_cache = pickle.load(open('cachehigh', 'rb'))
        n_cache = len(prev_cache)
    #else:
    #    n_cache = 0
    if len(U_stone.cachehigh) > n_cache:
        pickle.dump(U_stone.cachehigh, open('cachehigh', 'wb'))
    t1 = time.time()
    print("Finished with %d high quality U_stone.cache in %.2f seconds."% (len(U_stone.cachehigh), t1-t0))


def benchmark():
    my_stones = {(8,8),(8,9),(8,10),(8,11),(9,10),(11,12),(9,11),(9,12),(7,12)}
    last_move = (9,10)
    assert(i_win(my_stones, last_move)) == False
    t0 = time.time()
    n_repeat = 100000*10
    for _ in xrange(n_repeat):
        i_win(my_stones, last_move)
    t1 = time.time()
    print("--- %f ms per i_win() call ---" %((t1-t0)*1000/n_repeat))

def benchmark2():
    global board_size
    board_size = 15
    my_stones = {(8,8),(8,9),(8,10),(8,11),(9,10),(11,12),(9,11),(9,12),(7,12)}
    last_move = (8,11)
    blocking_stones = {(7,7),(7,9),(7,8),(7,10),(9,9),(10,10),(11,11),(12,11)}
    assert i_will_win(my_stones, blocking_stones, last_move) == True
    t0 = time.time()
    n_repeat = 100000 * 5
    for _ in xrange(n_repeat):
        i_will_win(my_stones, blocking_stones, last_move)
    t1 = time.time()
    print("--- %f ms per i_will_win() call ---" %((t1-t0)*1000/n_repeat))

def test():
    #ai_stones = {(8,7),(9,8),(10,9)}
    #player_stones = {(8,8),(8,9),(8,10),(8,11)}

    ai_stones = {(4,7),(4,10),(5,9),(6,9),(7,6),(8,8),(8,9)}
    player_stones = {(4,5),(5,6),(5,7),(6,7),(7,7),(7,8),(7,9)}
    #ai_stones = {(8,8),(5,9),(6,9)}
    #player_stones = {(7,9),(7,10),(7,11)}
    board = (player_stones, ai_stones)
    playing = 1
    last_move = (7,11)
    state = (board, last_move, playing, 15)
    t0 = time.time()
    print(strategy(state))
    t1 = time.time()
    print("--- %f s  ---" %(t1-t0))


def test3():
    my_stones = {(6,7),(7,8),(8,5),(8,7),(8,8),(8,9)}
    #my_stones = {(6,7),(7,8),(8,5),(8,7),(8,8),(8,9), (9,5)}
    #my_stones = {(6,7),(7,8),(8,5),(8,7),(8,8),(8,9), (6,10)}
    opponent_stones = {(5,6),(7,6),(7,7),(7,9),(8,6),(9,8)}
    board = (my_stones, opponent_stones)
    playing = 0
    last_move = (8,6)
    #playing = 1
    #last_move = (6,10)
    state = (board, last_move, playing, 15)
    global move_interest
    move_interest[6,9] = 1.0
    #move_interest[6,6] = 1.0
    print(strategy(state))

def board_show(stones):
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

def check():
    global board_size
    board_size = 15
    state = np.zeros(board_size**2, dtype=np.int32).reshape(board_size, board_size)
    # check if i_win() is working properly
    state[zip(*[(8,9), (8,11), (8,8), (8,10), (8,12)])] = 1
    assert i_win(state, (8,10), 1) == True
    state.fill(0)
    state[zip(*[(8,10), (9,11), (8,8), (9,12), (7,9), (10,9), (11,12), (11,13)])] = 1
    assert i_win(state, (10,12), 1) == True
    state.fill(0)
    state[zip(*[(8,10), (8,12), (8,8), (9,12), (7,9), (10,9), (11,12), (11,13)])] = 1
    assert i_win(state, (10,12), 1) == False
    # check if i_will_win() is working properly
    # o - x x X x - o
    state.fill(0)
    state[zip(*[(8,9), (8,11), (8,8)])] = 1
    state[zip(*[(8,6), (8,13)])] = -1
    assert i_will_win(state, (8, 10), 1) == True

    #
    state.fill(0)
    state[zip(*[(7,7), (7,8), (9,11)])] = 1
    state[zip(*[(6,8), (7,9)])] = -1
    print(state)
    assert i_will_win(state, (8,10), -1) == False
    ## o - x x X x o
    #assert i_will_win({(8,9), (8,11), (8,8)}, {(8,6), (8,12)}, (8,10)) == False
    ## o - x x X o
    ##         x
    ##
    ##         x
    ##         x
    #assert i_will_win({(8,9), (8,8), (9,10), (11,10), (12,10)}, {(8,6), (8,11)}, (8,10)) == False
    ## o - x x X x o
    ##         x
    ##
    ##         x
    ##         x
    #assert i_will_win({(8,9), (8,8), (9,10), (11,10), (12,10)}, {(8,6), (8,11)}, (8,10)) == False
    ## o - x x X x o
    ##       x
    ##
    ##   x
    ## x
    #assert i_will_win({ (8,8), (8,9), (8,11), (9,9), (11,7), (12,6)}, {(8,6), (8,12)}, (8,10)) == True
    ## | x x x X - x x x - - o
    #assert i_will_win({(8,1), (8,2), (8,0), (8,9), (8,7), (8,8)}, {(8,10)}, (8,3)) == False
    ## | x x - x X x x o
    #assert i_will_win({(8,1), (8,2), (8,4), (8,6), (8,7)}, {(8,8)}, (8,5)) == False
    ## | x x - x X - x x o
    #assert i_will_win({(8,1), (8,2), (8,4), (8,7), (8,8)}, {(8,9)}, (8,5)) == True
    ## | x x x - X - x x x o
    #assert i_will_win({(8,1), (8,2), (8,3), (8,7), (8,8), (8,9)}, {(8,10)}, (8,5)) == True
    ## | x - x X x - x o
    #assert i_will_win({(8,1), (8,3), (8,5), (8,7)}, {(8,8)}, (8,4)) == True

    #assert i_will_win({(8,8), (8,10), (9,9), (11,7), (11,9)}, {(7,7), (7,9), (8,7), (10,8), (11,8)}, (8,9)) == False
    print("All check passed!")

if __name__ == '__main__':
    import time
    check()
    #test3()
    #benchmark()
    #benchmark2()
