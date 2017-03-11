#!/usr/bin/env python

import itertools, time, copy
import collections, random
import os, pickle
import numba

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
    global board_size
    board, last_move, playing, board_size = state
    other_player = int(not playing)
    my_stones = board[playing]
    opponent_stones = board[other_player]
    # put the first stone in the center if it's the start of the game
    if len(my_stones) is 0 and len(opponent_stones) is 0:
        center = int((board_size+1)/2)
        return (center, center)

    if not hasattr(U_stone, 'cache'):
        initialize()

    q_state = (my_stones, opponent_stones)
    return best_action_q(q_state, -1)[0]

def best_action_q(q_state, level):
    "Return the optimal action for a state"
    max_q = -1
    best_move = None
    all_stones = set(q_state[0] | q_state[1])
    for last_move in available_positions(all_stones):
        q = Q_stone(q_state, last_move, level=level)
        if q == 1.0:
            return last_move, 1.0
        elif q > max_q:
            max_q = q
            best_move = last_move
    return best_move, max_q

@numba.jit(nopython=True,nogil=True)
def available_positions(all_stones):
    positions = set()
    for x in range(1, board_size+1):
        for y in range(1, board_size+1):
            stone = (x,y)
            if stone not in all_stones and near_any_stone(stone, all_stones):
                positions.add(stone)
    return positions

@numba.jit(nopython=True,nogil=True)
def near_any_stone(last_move, all_stones):
    r1, c1 = last_move
    for r2, c2 in all_stones:
        if abs(r2-r1) < 2 and abs(c2-c1) < 2:
            return True
    return False


def Q_stone(q_state, last_move, level):
    my_stones, opponent_stones = q_state
    my_stones_new = set(my_stones)
    my_stones_new.add(last_move)
    u_state = (my_stones_new, opponent_stones, last_move)
    return U_stone(u_state, level=level)

def U_stone(u_state, level):
    my_stones, opponent_stones, last_move = u_state

    # try to find existing answer from the high quality cache first
    key = (frozenset(my_stones), frozenset(opponent_stones))
    #print len(U_stone.cache)
    try:
        return U_stone.cache[key]
    except:
        pass

    #if level == 1:
    #    print my_stones, opponent_stones
    if i_win(my_stones, last_move):
        result = 1.0
        U_stone.cache[key] = result
    elif level > 2:
        result = 0.1 + 0.5 * random.random()
    else:
        # go to the next player
        q_state = (opponent_stones, my_stones)
        best_move, max_q = best_action_q(q_state, level+1)
        result = 1.0 - max_q
        if result == 0.0:
            U_stone.cache[key] = result

    return result

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
    if len(my_stones) < 5: return False
    r, c = last_move
    # find any nearby stone
    nearby_stones = set(( (r-1,c-1), (r-1,c), (r-1,c+1), (r,c-1), (r,c+1), (r+1,c-1), (r+1,c), (r+1,c+1) ))
    nearby_stones &= my_stones
    for nearby_s in nearby_stones:
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
        for i in range(1,5):
            ext_stone = (r-dx*i, c-dy*i)
            if ext_stone in my_stones:
                line_length += 1
            else:
                break
        if line_length >= 5:
            return True
    return False

def initialize():
    global n_exist
    # load the cache
    if os.path.exists('aicache'):
        U_stone.cache = pickle.load( open('aicache',"rb") )
        n_exist = len(U_stone.cache)
        print('Successfully loaded %d aicache data'%n_exist)
    elif os.path.exists('aicache.gz'):
        import gzip
        U_stone.cache = pickle.load( gzip.open('aicache.gz',"rb") )
        n_exist = len(U_stone.cache)
        print('Successfully loaded %d conpressed high quality cache data'%n_exist)
    else:
        U_stone.cache = dict()
        n_exist = 0
        print("aicache is not found, I will be learning from the beginning!")

def finish():
    if len(U_stone.cache) <= n_exist: return
    if os.path.exists('aicache.gz'):
        pickle.dump( U_stone.cache, gzip.open("aicache.gz","wb") )
    else:
        pickle.dump( U_stone.cache, open("aicache","wb") )
    print('Successfully updated U_stone.cache data with %d states'%len(U_stone.cache))



def training():
    pass


if __name__ == '__main__':
    training()
