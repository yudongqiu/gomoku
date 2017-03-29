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

    initialize()

    global board_size

    board, last_move, playing, board_size = state
    other_player = int(not playing)
    my_stones = board[playing]
    opponent_stones = board[other_player]
    # put the first stone in the center if it's the start of the game
    center = int((board_size+1)/2)
    if last_move is None:
        return (center, center)
    elif len(my_stones) == 0:
        # Assuming the first stone was put on the center
        return random.choice([(center-1, center+1), (center-1,center)])
        #return random.choice(list(nearby_avail_positions(last_move, opponent_stones)))

    state = (my_stones, opponent_stones)




    global lv1_win_rates
    if last_move in lv1_win_rates:
        print("Calculated Move: %.3f" %lv1_win_rates[last_move])
    else:
        print("Didn't know this move!")
    lv1_win_rates = dict()

    global move_interest
    global lv1_interest_matrices
    lv1_interest_matrices = dict() # clean the saved lv1 interest matrices


    #U_stone.cache = U_stone.cachehigh.copy()
    U_stone.cache = dict()
    alpha = -1.0
    beta = 2.0
    best_move, best_q = best_action_q(state, last_move, move_interest, alpha, beta, True, 0)


    # If the win rate of this turn is lower than my last turn, means that my previous estimate was wrong:
    #if best_q < last_q:
    #    U_stone.cache[]

    if best_q == 0:
        return (0,0) # admit defeat if I'm losing

    # update the global move_interest matrix with the lv1
    # the old global move_interest matrix is discarded here because it's out-of-date
    #if len(lv1_interest_matrices) > 0:
    #    move_interest = lv1_interest_matrices[best_move]
    try:
        move_interest = lv1_interest_matrices[best_move]
    except:
        pass
        #move_interest *= 0.9

    return best_move

move_interest = np.zeros((board_size+1)**2).reshape(board_size+1, board_size+1)
lv1_interest_matrices = dict()
lv1_win_rates = dict()

level_max_n = [120, 50, 15, 10, 8, 7, 6, 1, 1, 1, 1, 1, 1, 1, 1]
#level_max_n = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
def best_action_q(state, last_move, move_interest, alpha, beta, maximizing_player, level):
    "Return the optimal action for a state"
    global lv1_win_rates
    global lv1_interest_matrices

    best_move = (0,0) # admit defeat if all moves have 0 win rate
    possible_moves = available_positions(state, 2)
    if len(possible_moves) == 1:
        current_move = possible_moves.pop()
        #if level == 0: # We have no choice here
        #    return current_move, 0.5
        q = Q_stone(state, current_move, move_interest, alpha, beta, maximizing_player, level)
        # record the opponent's next move's q
        if level <= 1 and maximizing_player is False:
            lv1_win_rates[current_move] = q
        return current_move, q

    local_interest = move_interest.copy() * 0.9 # current move_interest to be passed and modified by next level

    threat_moves = find_threat_positions(state)
    local_interest[zip(*threat_moves)] = 0.99

    n_candidates = min(len(possible_moves), level_max_n[level] + len(threat_moves))

    if maximizing_player:
        max_q = 0.0
        for _ in xrange(n_candidates):
            possible_moves = sorted(possible_moves, key=lambda m: local_interest[m])
            current_move = possible_moves.pop()
            q = Q_stone(state, current_move, local_interest, alpha, beta, maximizing_player, level+1)
            if q > move_interest[current_move]: # update the parent's interest matrix
                move_interest[current_move] = q
            #if q > local_interest[current_move]: # update the local interest matrix? seems unnecessary because the child node will do this for me
            #    local_interest[current_move] = q
            if q > alpha: alpha = q
            if q > max_q:
                if level == 0:
                    print(current_move, q, "interest", local_interest[current_move])
                max_q = q
                best_move = current_move
            if q == 1.0 or beta <= alpha:
                break
        best_q = max_q
    else:
        min_q = 1.0
        for _ in xrange(n_candidates):
            possible_moves = sorted(possible_moves, key=lambda m: local_interest[m])
            current_move = possible_moves.pop()
            q = Q_stone(state, current_move, local_interest, alpha, beta, maximizing_player, level+1)
            if (1 - q) > move_interest[current_move]:
                move_interest[current_move] = (1 - q)
            #if (1 - q) > local_interest[current_move]:
            #    local_interest[current_move] = (1 - q)
            if q < beta: beta = q
            if q < min_q:
                min_q = q
                best_move = current_move
            if level <= 1:
                lv1_win_rates[current_move] = q
            if q == 0.0 or beta <= alpha:
                break
        best_q = min_q
        if level == 1: #if the opponent can choose where to move
            # save lv 1 local_interest for each lv0 moves
            lv1_interest_matrices[last_move] = local_interest
    # if lv1_interest_matrices is empty, means that there is no choice for the opponent, we will use our lv0 local_interest to update the global move_interest
    if level is 0 and len(lv1_interest_matrices) == 0:
        # update the global interest matrix with level 0 local_interest (modified by all lv1 calls)
        np.copyto(move_interest, local_interest)
    return best_move, best_q

@numba.jit(nopython=True, nogil=True)
def available_positions(state, dist=1):
    my_stones, opponent_stones = state
    positions = set()
    limiting = False # limiting positions if someone is winning
    for x in range(1, board_size+1):
        for y in range(1, board_size+1):
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

@numba.jit(nopython=True, nogil=True)
def find_threat_positions(state):
    my_stones, opponent_stones = state
    threats = []
    #my_threats = []
    #opponent_threats = []
    for x in range(1, board_size+1):
        for y in range(1, board_size+1):
            stone = (x,y)
            if stone not in my_stones and stone not in opponent_stones:
                if i_will_win(my_stones, opponent_stones, stone) or i_will_win(opponent_stones, my_stones, stone):
                    threats.append(stone)
                #if i_will_win(my_stones, opponent_stones, stone):
                #    my_threats.append(stone)
                #if i_will_win(opponent_stones, my_stones, stone):
                #    opponent_threats.append(stone)
    return threats
    #if len(my_threats) > 0:
    #    return my_threats
    #else:
    #    return opponent_threats


@numba.jit(nopython=True, nogil=True)
def nearby_avail_positions(this_stone, all_stones):
    """ Find available positions on the board that are adjacent to this_stone """
    r, c = this_stone
    result = set()
    nearby_pos = {(r-1,c-1), (r-1,c), (r-1,c+1), (r,c-1), (r,c+1), (r+1,c-1), (r+1,c), (r+1,c+1)}
    for stone in nearby_pos:
        if 0 < stone[0] <= board_size and 0 < stone[1] <= board_size and stone not in all_stones:
            result.add(stone)
    return result

@numba.jit(nopython=True, nogil=True)
def available_positions2(state):
    my_stones, opponent_stones = state
    positions = set()
    for x in range(1, board_size+1):
        for y in range(1, board_size+1):
            stone = (x,y)
            if stone not in my_stones and stone not in opponent_stones and near_any_stone(stone, state, 1):
                positions.add(stone)
    return positions

#@numba.jit(nopython=True, nogil=True)
def Q_stone(state, current_move, move_interest, alpha, beta, maximizing_player, level):
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
    return U_stone(state, current_move, move_interest, alpha, beta, maximizing_player, level)

#@numba.jit(nopython=True, nogil=True)
def U_stone(state, last_move, move_interest, alpha, beta, maximizing_player, level):
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

    MC_start_level = 10
    #if maximizing_player and i_win(my_stones, last_move):
    if maximizing_player is True and i_will_win(my_stones, opponent_stones, last_move):
        return 1.0
    #elif not maximizing_player and i_win(opponent_stones, last_move):
    elif maximizing_player is False and i_will_win(opponent_stones, my_stones, last_move):
        return 0.0
    #elif level == MC_start_level:
    #    #return cached_MC(state, maximizing_player, 19, 20)
    #    this_U = MC_estimate_U(state, maximizing_player, 19, 20)
    #    next_move, next_U = best_action_q(state, last_move, alpha, beta, not maximizing_player, level)
    #    if next_U == 0.0 or next_U == 1.0:
    #        result = next_U
    #    else:
    #        result = 0.5 * (this_U + next_U)
    elif level >= MC_start_level:
        result = MC_estimate_U(state, maximizing_player, 19, 20)
    else:
        best_move, best_q = best_action_q(state, last_move, move_interest, alpha, beta, not maximizing_player, level)
        #if best_q == 1.0 or best_q == 0.0:
        #    if maximizing_player is True:
        #        U_stone.cachehigh[key] = best_q
        #    else:
        #        U_stone.cachehigh[key] = 1.0 - best_q
        result = best_q

    if maximizing_player:
        cached_result = result
    else:
        cached_result = 1.0 - result

    #if cached_result == 1.0 or cached_result == 0.0:# or level <= MC_start_level - 5:
    #    # save the high quality
    #    U_stone.cachehigh[key] = cached_result
    U_stone.cache[key] = cached_result

    return result

@numba.jit(nopython=True, nogil=True)
def MC_estimate_U(state, maximizing_player, n_MC, max_steps):
    """ Randomly put stones until the game ends, estimate the U based on number of games won. """
    my_stones, opponent_stones = state
    n_win = 0.5
    #if len(all_stones) < 4: return 0.5
    all_possible_moves = available_positions2(state)
    for _ in range(n_MC):
        # pool of all available positions
        current_possible_moves = list(all_possible_moves)
        current_possible_moves_set = all_possible_moves.copy()
        current_my_stone = my_stones.copy()
        current_opponent_stone = opponent_stones.copy()
        winning_player = int(not maximizing_player) # 1 if it's my turn, 0 if it's opponent's turn
        max_i_move = len(current_possible_moves) - 1
        i_step = 0
        while True:
            # choose a random stone from the pool
            i_move = random.randint(0, max_i_move)
            random_move = current_possible_moves.pop(i_move)
            #random_move = current_possible_moves[i_move]
            current_possible_moves_set.remove(random_move)

            max_i_move -= 1
            # place that stone for the current player
            if winning_player:
                current_stones = current_my_stone
                blocking_stones = current_opponent_stone
            else:
                current_stones = current_opponent_stone
                blocking_stones = current_my_stone
            current_stones.add(random_move)

            r, c = random_move
            nearby_stones = ((r-1,c-1), (r+1,c+1), (r-1,c), (r+1,c), (r-1,c+1), (r+1,c-1), (r,c-1), (r,c+1))
            winning = False
            tested = -1 # if the next stone is already tested (opposite side)
            for i_ns in range(8):
                ns = nearby_stones[i_ns]
                if ns in current_stones:
                    if i_ns == tested:
                        tested = -1
                    else:
                        if i_ns % 2 == 0:
                            tested = i_ns + 1 # skip the next nearby stone
                        nr, nc = ns
                        dx, dy = nr-r, nc-c
                        line_length = 2 # last_move and nearby_s
                        # try to extend in this direction
                        for i in range(1,4):
                            ext_stone = (nr+dx*i, nc+dy*i)
                            if ext_stone in current_stones:
                                line_length += 1
                            elif ext_stone not in blocking_stones: # potential win
                                line_length += 0.5
                                break
                            else:
                                break
                        # try to extend in the opposite direction
                        for i in range(1,4):
                            ext_stone = (r-dx*i, c-dy*i)
                            if ext_stone in current_stones:
                                line_length += 1
                            elif ext_stone not in blocking_stones: # potential win
                                line_length += 0.5
                                break
                            else:
                                break
                        if line_length >= 5:
                            winning = True
                            break
                else:
                    if ns not in current_possible_moves_set:
                        current_possible_moves_set.add(ns)
                        current_possible_moves.append(ns)
                        max_i_move += 1

            # check if game ends
            if winning:
                n_win += winning_player
                break
            if max_i_move == 0 or i_step > max_steps: # this is a tie
                n_win += 0.5
                break
            # goto next player
            winning_player = int(not winning_player)
            i_step += 1
    return n_win / (n_MC+1)



@numba.jit(nopython=True,nogil=True)
def i_win_old(my_stones, last_move):
    """ Return true if I just got 5-in-a-row with last_move """
    if len(my_stones) < 4: return False
    r, c = last_move
    # find any nearby stone
    nearby_stones = ((r-1,c-1), (r+1,c+1), (r-1,c), (r+1,c), (r-1,c+1), (r+1,c-1), (r,c-1), (r,c+1))
    #nearby_stones &= my_stones
    skip_next = False
    #for i_ns in range(8):
    i_ns = 0
    while True:
        if i_ns >= 8:
            break
        nearby_s = nearby_stones[i_ns]
        if nearby_s not in my_stones:
            i_ns += 1
            continue
        line_length = 2 # last_move and nearby_s
        nr, nc = nearby_s
        dx, dy = nr-r, nc-c
        # try to extend in this direction
        for i in range(1,4):
            ext_stone = (nr+dx*i, nc+dy*i)
            if ext_stone in my_stones:
                line_length += 1
            else:
                break
        if line_length is 5:
            return True
        # try to extend in the opposite direction
        for i in range(1, 6-line_length):
            ext_stone = (r-dx*i, c-dy*i)
            if ext_stone in my_stones:
                line_length += 1
            else:
                break
        if line_length is 5:
            return True
        i_ns += (2 - i_ns % 2) # the next one on the opposite side is already explored
    return False

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
            if ext_r < 1 or ext_r > board_size or ext_c < 1 or ext_c > board_size: break
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
            if ext_r < 1 or ext_r > board_size or ext_c < 1 or ext_c > board_size: break
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
    if not hasattr(U_stone, 'cachehigh'):
        t0 = time.time()
        if os.path.exists("cachehigh"):
            U_stone.cachehigh = pickle.load(open('cachehigh', 'rb'))
        else:
            U_stone.cachehigh = dict()
        global n_cache
        n_cache = len(U_stone.cachehigh)
        t1 = time.time()
        #print("Initialized with %d high quality U_stone.cache in %.2f seconds." % (n_cache, t1-t0))

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
    #test3()
    #benchmark()
    #benchmark2()
