#!/usr/bin/env python

import itertools, time, copy
import collections, random
import os, pickle
import numba
import numpy as np

def memo(f):
    """Decorator that caches the return value for each call to f(args).
    Then when called again with same args, we can just look it up."""
    cache = {}
    def _f(*args):
        try:
            return cache[args]
        except KeyError:
            cache[args] = result = f(*args)
            return result
        except TypeError:
            # some element of args refuses to be a dict key
            return f(*args)
    _f.cache = cache
    return _f

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
    if last_move is None:
        center = int((board_size+1)/2)
        return (center, center)
    #elif len(my_stones) == 0:
    #    return random.choice(list(nearby_avail_positions(last_move, opponent_stones)))

    state = (my_stones, opponent_stones)

    alpha = -1.0
    beta = 2.0
    global move_interest

    try:
        move_interest *= 0.8
    except:
        move_interest = np.zeros((board_size+1)**2).reshape(board_size+1, board_size+1)

    U_stone.cache = U_stone.cachehigh.copy()
    #U_stone.cache = dict()
    best_move, best_q = best_action_q(state, last_move, alpha, beta, True, 0)

    return best_move


def best_action_q(state, last_move, alpha, beta, maximizing_player, level):
    "Return the optimal action for a state"
    #global estimate_cache, best_lv1, pruned, computed
    #my_stones, opponent_stones = state
    best_move = (0,0) # admit defeat if all moves have 0 win rate
    #all_stones = my_stones | opponent_stones
    possible_moves = available_positions(state, 2-int(level>1))
    #if level == 0:
    #    print(len(possible_moves))
    n_candidates = len(possible_moves)
    if level > 1 and n_candidates > 10:
        n_candidates = 10
        #n_candidates = int(10 + n_candidates/10)
        #n_candidates = 15 - level
    if maximizing_player:
        max_q = 0.0
        for _ in xrange(n_candidates):
            possible_moves = sorted(possible_moves, key=lambda m: move_interest[m])
            current_move = possible_moves.pop()
            q = Q_stone(state, current_move, alpha, beta, maximizing_player, level+1)
            if level == 0:
                print current_move, q, "interest", move_interest[current_move]
            if q * (0.8**level) > move_interest[current_move]:
                move_interest[current_move] = q * (0.8**level)
            if q > alpha: alpha = q
            if q > max_q:
                max_q = q
                best_move = current_move
            if q == 1.0 or beta <= alpha:
                break
        best_q = max_q
    else:
        min_q = 1.0
        for _ in xrange(n_candidates):
            possible_moves = sorted(possible_moves, key=lambda m: move_interest[m])
            current_move = possible_moves.pop()
            q = Q_stone(state, current_move, alpha, beta, maximizing_player, level+1)
            if (1 - q) * (0.8**(level-1)) > move_interest[current_move]:
                move_interest[current_move] = (1 - q) * (0.8**(level-1))
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

@numba.jit(nopython=True,nogil=True)
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
def nearby_stones(this_stone):
    """ Find available positions on the board that are adjacent to this_stone """
    r, c = this_stone
    result = set()
    nearby_pos = {(r-1,c-1), (r-1,c), (r-1,c+1), (r,c-1), (r,c+1), (r+1,c-1), (r+1,c), (r+1,c+1)}
    for stone in nearby_pos:
        if 0 < stone[0] <= board_size and 0 < stone[1] <= board_size:
            result.add(stone)
    return result

#@numba.jit(nopython=True, nogil=True)
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

#@numba.jit(nopython=True, nogil=True)
def U_stone(state, last_move, alpha, beta, maximizing_player, level):
    my_stones, opponent_stones = state
    if maximizing_player: # put the stones of the current player first
        key = (frozenset(my_stones), frozenset(opponent_stones))
    else:
        key = (frozenset(opponent_stones), frozenset(my_stones))
    try:
        return U_stone.cache[key]
    except:
        pass

    MC_start_level = 7
    if maximizing_player and i_win(my_stones, last_move):
        result = 1.0
    elif not maximizing_player and i_win(opponent_stones, last_move):
        result = 0.0
    elif level >= MC_start_level:
        #return cached_MC(state, maximizing_player, 19, 20)
        result = MC_estimate_U(state, maximizing_player, 19, 20)
    else:
        best_move, best_q = best_action_q(state, last_move, alpha, beta, not maximizing_player, level)
        result = best_q

    if level <= 2 or result == 1.0 or result == 0.0:
        # save the high quality
        U_stone.cachehigh[key] = result
    U_stone.cache[key] = result

    return result


def cached_MC(state, maximizing_player, n_MC, max_steps):
    if not hasattr(cached_MC, 'cache'):
        cached_MC.cache = dict()
    key = (frozenset(state[0]), frozenset(state[1]))
    try:
        return cached_MC.cache[key]
    except:
        cached_MC.cache[key] = result = MC_estimate_U(state, maximizing_player, n_MC, max_steps)
        return result

@numba.jit(nopython=True, nogil=True)
def MC_estimate_U(state, maximizing_player, n_MC, max_steps):
    """ Randomly put stones until the game ends, estimate the U based on number of games won. """
    my_stones, opponent_stones = state
    n_win = 0.0
    #if len(all_stones) < 4: return 0.5
    all_possible_moves = available_positions(state)
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
    return n_win / n_MC



#@numba.jit(nopython=True, nogil=True)
#def min_stone_dist(last_move, all_stones):
#    r1, c1 = last_move
#    min_dist = board_size
#    for r2, c2 in all_stones:
#        dist = max(abs(r1-r2), abs(c1-c2))
#        if dist < min_dist:
#            min_dist = dist
#    return min_dist



def min_stone_dist(last_move, all_stones):
    return min(stone_dist(last_move, stone) for stone in all_stones)

@memo
def stone_dist(stone1, stone2):
    r1, c1 = stone1
    r2, c2 = stone2
    return max(abs(r1-r2), abs(c1-c2))

@numba.jit(nopython=True,nogil=True)
def i_win(my_stones, last_move):
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
        # try to extend in the opposite direction
        for i in range(1,4):
            ext_stone = (r-dx*i, c-dy*i)
            if ext_stone in my_stones:
                line_length += 1
            else:
                break
        if line_length >= 5:
            return True
        i_ns += (2 - i_ns % 2) # the next one on the opposite side is already explored
    return False

def initialize():
    if not hasattr(U_stone, 'cachehigh'):
        if os.path.exists("cachehigh"):
            U_stone.cachehigh = pickle.load(open('cachehigh', 'rb'))
        else:
            U_stone.cachehigh = dict()
        global n_cache
        n_cache = len(U_stone.cachehigh)
        print("Initialized with %d high quality U_stone.cache" % n_cache)

def finish():
    if len(U_stone.cachehigh) > n_cache:
        pickle.dump(U_stone.cachehigh, open('cachehigh', 'wb'))
    print("Finished with %d high quality U_stone.cache."%len(U_stone.cachehigh))


def benchmark():
    assert i_win({(8, 9), (8, 11), (8, 8), (8, 10), (8, 12)}, (8,10)) == True
    my_stones = {(8,8),(8,9),(8,10),(8,11),(9,10),(11,12),(9,11),(9,12),(7,12)}
    last_move = (9,10)

    t0 = time.time()
    n_repeat = 100000*10
    for _ in xrange(n_repeat):
        i_win(my_stones, last_move)
    t1 = time.time()
    print("--- %f ms per i_win() call ---" %((t1-t0)*1000/n_repeat))

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
    print strategy(state)
    t1 = time.time()
    print("--- %f s  ---" %(t1-t0))

def test2():
    my_stones = {(6,7),(7,8),(8,8),(8,9)}
    opponent_stones = {(5,7),(6,8),(7,9),(8,10)}
    global board_size
    board_size = 15
    state = (my_stones, opponent_stones)
    print i_win(opponent_stones, (4,6))
    print available_positions(state)

if __name__ == '__main__':
    import time
    test2()
    #benchmark()
