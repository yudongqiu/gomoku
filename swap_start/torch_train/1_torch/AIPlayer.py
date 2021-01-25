#!/usr/bin/env python

from __future__ import print_function, division
import itertools, time, copy
import collections, random
import os, pickle
import numba
from numba import cuda
import numpy as np

board_size = 15
show_q = False

class AIPlayer:
    def __init__(self, name, model, level=0):
        self.name = name
        self.model = model
        self.level = level
        self.learndata = dict()
        self.opponent = None
        self.all_interest_states = np.zeros(board_size**4 * 3, dtype=np.float32).reshape(board_size**2, 3, board_size, board_size)
        self.move_interest_values = np.zeros(board_size**2, dtype=np.float32).reshape(board_size,board_size)
        self.reset()
        self.reset_cache()

    def reset(self):
        """ Reset before a new game """
        self.hist_states = []
        self.surprised = False
        self.started_from_beginning = True

    def reset_cache(self):
        """ Reset cache before using new model """
        self.cache = LeveledCache(maxlevel=self.level, maxsize=1000000)

    def strategy(self, board_state):
        """ AI's strategy 
        Information provided to you:
        board_state = (board, last_move, playing, board_size)
        board = (x_stones, o_stones)
        stones is a set contains positions of one player's stones. e.g.
            x_stones = {(8,8), (8,9), (8,10), (8,11)}
        playing = 0|1, the current player's index

        Your strategy will return a position code for the next stone, e.g. (8,7)
        """
        # load input board_state
        board, last_move, playing, board_size = board_state
        self.playing_white = bool(playing)
        # build new state representation
        state = np.zeros(board_size**2, dtype=np.int8).reshape(board_size, board_size)
        # put black stones, update index
        for br, bc in board[0]:
            state[br-1,bc-1] = 1
        # put white stones, update index
        for wr, wc in board[1]:
            state[wr-1,wc-1] = -1
        # update index 1 -> 0 for last_move
        last_move = (last_move[0]-1, last_move[1]-1)
        # prepare input for best_action_q
        alpha = -2.0
        beta = 2.0
        empty_spots_left = board_size**2 - len(board[0]) - len(board[1])
        # predict next best action and q
        player = -1 if self.playing_white else 1
        best_move, best_q = self.best_action_q(state.copy(), empty_spots_left, alpha, beta, player, level=0)
        # save the winrate and the state
        self.update_if_game_finish(state, best_move, best_q, player)
        # return the best move
        return (best_move[0]+1, best_move[1]+1)

    def best_action_q(self, state, empty_spots_left, alpha, beta, player, level=0):
        """ 
        Get the optimal action for a state and the predicted win rate for player

        Inputs
        ------
        state: np.ndarray of shape (15, 15)
            The current game state in a matrix. 1 = black, -1 = white, 0 = empty
        empty_spots_left: int
            How many empty spots are left, easy to keep track
        alpha: float
            Current alpha value in alpha-beta pruning, the running min of the max win rate
        beta: float
            Current beta value in alpha-beta pruning, the running max of the min win rate
        player: int
            The current player. 1 is black, -1 is white

        Returns
        -------
        best_move: tuple(int, int)
            The best move on the board, given by (r, c)
        best_q: float or None
            The value the best move. 1.0 means 100% win, -1.0 means 100% lose, 0 means draw
        """
        if empty_spots_left == 0: # Board filled up, it's a tie
            return (0,0), 0.0
        verbose = False
        n_moves = 40 if empty_spots_left > 200 else 20
        self.move_interest_values.fill(0) # reuse the same array to save init cost
        self.move_interest_values[4:11, 4:11] = 5.0 # manually assign higher interest in middle
        interested_moves = find_interesting_moves(state, empty_spots_left, self.move_interest_values, player, n_moves, verbose)
        #best_move = (-1,-1) # admit defeat if all moves have 0 win rate
        best_move = (interested_moves[0,0], interested_moves[0,1]) # continue to play even I'm losing
        # if there is only one move to place, directly return that move, use same level
        if len(interested_moves) == 1:
            # check if this move is known
            move, move_q, unknown_moves, unknown_move_ids = self.check_known(state, interested_moves, player, level)
            if move != None:
                best_q = move_q
            else:
                best_q = self.next_iter_winrate(state, empty_spots_left, best_move, alpha, beta, player, level)
            return best_move, best_q
        # if there are multiple moves to evaluate, check cache first
        best_move, max_q, unknown_moves, unknown_move_ids = self.check_known(state, interested_moves, player, level)
        if len(unknown_moves) > 0:
            # for unknown moves, if level has reached, evaluate with DNN model
            if level >= self.level:
                dnn_q_array = self.dnn_evaluate(state, unknown_moves, player)
                # store the values in cache
                for move_id, dnn_q in zip(unknown_move_ids, dnn_q_array):
                    self.cache.set(move_id, dnn_q, level)
                # find the best move from tf results
                dnn_best_move_idx = np.argmax(dnn_q_array)
                dnn_max_q = dnn_q_array[dnn_best_move_idx]
                # compare the tf results with cached results
                if dnn_max_q > max_q:
                    max_q = dnn_max_q
                    best_move = unknown_moves[dnn_best_move_idx]
            else:
                # if level has not reached yet, go deeper to the next level
                for move, move_id in zip(unknown_moves, unknown_move_ids):
                    q = self.next_iter_winrate(state, empty_spots_left, move, alpha, beta, player, level+1)
                    # store the result in cache
                    self.cache.set(move_id, q, level+1)
                    if q > max_q:
                        max_q = q
                        best_move = move
                    if max_q >= 1.0:
                        # early return
                        break
        return best_move, max_q

    def next_iter_winrate(self, state, empty_spots_left, current_move, alpha, beta, player, level):
        """Execute the step of the player, then return the winrate by computing next step"""
        # update the stone down
        state[current_move] = player
        # known moves were handled already, here we evaluate opponents winrate
        opponent_best_move, opponent_best_q = self.best_action_q(state, empty_spots_left-1, alpha, beta, -player, level)
        # recover state
        state[current_move] = 0
        # my winrate is opposite of opponents
        return -opponent_best_q

    def check_known(self, state, interested_moves, player, level):
        """
        Check which move in interested moves is known, using cache and ending condition
        """
        max_q = -100
        best_move = None
        unknown_moves = []
        unknown_move_ids = []
        for move in interested_moves:
            this_move = (move[0], move[1])
            assert state[this_move] == 0 # interest move should be empty here
            # put down this move
            state[this_move] = player
            # if I lost, no need to compute others
            if i_lost(state, player):
                # restore state 
                state[this_move] = 0
                # early return
                best_move = this_move
                max_q = -1.0
                unknown_moves = []
                unknown_move_ids = []
                break
            # compute cache key
            this_state_id = state.tobytes()
            # check if its cached
            q = self.cache.get(this_state_id, level)
            # if not cached, check if I will win
            if q is None:
                if i_will_win(state, this_move, player):
                    q = 1.0
            if q is not None:
                # early return when found winning move
                if q == 1.0:
                    # restore state 
                    state[this_move] = 0
                    # early return
                    best_move = this_move
                    max_q = 1.0
                    unknown_moves = []
                    unknown_move_ids = []
                    break
                # compare with running max, update
                elif q > max_q:
                    max_q = q
                    best_move = this_move
            else:
                # q is not known
                unknown_moves.append(this_move)
                unknown_move_ids.append(this_state_id)
            # restore state 
            state[this_move] = 0
        return best_move, max_q, unknown_moves, unknown_move_ids

    
    def dnn_evaluate(self, state, dnn_moves, player):
        n_dnn = len(dnn_moves)
        if n_dnn > 0:
            all_interest_states = self.all_interest_states[:n_dnn] # we only need a slice of the big array
            all_interest_states[:,0,:,:] = (state == player) # player's stones
            all_interest_states[:,1,:,:] = (state == -player) # opponent stones
            all_interest_states[:,2,:,:] = 1 if player == 1 else 0 # if player is black, set 1 else 0
            for i,current_move in enumerate(dnn_moves):
                ci, cj = current_move
                all_interest_states[i,0,ci,cj] = 1 # put current move down
            predict_y = self.model.predict(all_interest_states)
            return predict_y.ravel()
        else:
            return []

    def update_if_game_finish(self, state, best_move, best_q, player):
        # put down this step and record learn data
        # state[best_move] = player
        # state_id = state.tobytes()
        # self.learndata[state_id] = [state.copy(), best_q, 1]
        # state[best_move] = 0

        # store learn data for oppoenent, this helps improve the data
        state_id = state.tobytes()
        # if self.playing_white == False and best_q == -1.0:
            # import IPython; IPython.embed()

        self.opponent.learndata[state_id] = [state.copy(), -best_q, 1]

        # # record the history states
        # self.hist_states.append(opponent_state_id)
        # # check if game finish
        # state[best_move] = 1
        # game_result = None
        # new_u = 0
        # if i_win(state, best_move, 1):
        #     new_u = -1.0
        #     game_result = 'win'
        # elif i_lost(state, 1):
        #     new_u = 1.0
        #     game_result = 'lose'
        # elif empty_spots_left <= 2:
        #     new_u = 0
        #     game_result = 'draw'
        # if game_result and self.started_from_beginning is True:
        #     discount = 0.9
        #     for opponent_state_id in self.hist_states[::-1]:
        #         st, u, n_visited = self.opponent.learndata[opponent_state_id]
        #         n_visited += 1
        #         new_u = u + discount * (new_u - u) / n_visited**0.5 # this is the learning rate
        #         # surprise
        #         if (game_result == 'win' and new_u > 0.1) or (game_result == 'lose' and new_u < -0.1):
        #             self.surprised = True
        #         self.opponent.learndata[opponent_state_id] = (st, new_u, n_visited)
        #         print(f"Updated U from {u:9.6f} to {new_u:9.6f} [{n_visited}]")
        #     print(f"{self.name}: Updated win rate of {len(self.hist_states)} states")
        #     self.started_from_beginning = False # we only update once            




















# Below are utility functions

@numba.jit(nopython=True, nogil=True, cache=True)
def find_interesting_moves(state, empty_spots_left, move_interest_values, player, n_moves, verbose=False):
    """ Look at state and find the interesing n_move moves.
    input:
    -------
    state: numpy.array board_size x board_size, 1=black, -1=white, 0=empty
    empty_spots_left: number of empty spots on the board
    player: current player to find interesting moves, 1=black, -1=white
    n_moves: int, desired number of interesing moves

    output:
    -------
    interested_moves: numpy.array final_n_moves x 2
        *note : final_n_moves = 1 if limited
        *       else final_n_moves = n_moves + number of length-4 moves
        *note2: final_n_moves will not exceed empty_spots_left


    #suggested_n_moves: suggested number of moves to
    """
    force_to_block = False
    exist_will_win_move = False
    directions = ((1,1), (1,0), (0,1), (1,-1))
    final_single_move = np.zeros(2, dtype=np.int64).reshape(1,2) # for returning the single move
    for r in range(board_size):
        for c in range(board_size):
            if state[r,c] != 0: continue
            interest_value = 10 # as long as it's a valid point, this is for avoiding the taken spaces
            my_hard_4 = 0
            for dr, dc in directions:
                my_line_length = 1 # last_move
                opponent_line_length = 1
                # try to extend in the positive direction (max 5 times to check overline)
                ext_r = r
                ext_c = c
                skipped_1 = 0
                my_blocked = False
                opponent_blocked = False
                for i in range(5):
                    ext_r += dr
                    ext_c += dc
                    if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                        break
                    elif state[ext_r, ext_c] == player:
                        if my_blocked == True:
                            break
                        else:
                            my_line_length += 1
                            opponent_blocked = True
                    elif state[ext_r, ext_c] == -player:
                        if opponent_blocked == True:
                            break
                        else:
                            opponent_line_length += 1
                            my_blocked = True
                    elif skipped_1 == 0:
                        skipped_1 = i + 1 # allow one skip and record the position of the skip
                    else:
                        # peek at the next one and if it might be useful, add some interest
                        if ((state[ext_r+dr, ext_c+dc] == player) and (my_blocked == False)) or ((state[ext_r+dr, ext_c+dc] == -player) and (opponent_blocked == False)):
                            interest_value += 15
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
                    if my_blocked == False:
                        my_line_length_back = skipped_1
                        opponent_line_length_back = 1
                        forward_my_open = True
                    else:
                        my_line_length_back = 1
                        opponent_line_length_back = skipped_1
                        forward_opponent_open = True
                my_line_length_no_skip = my_line_length_back
                opponent_line_length_no_skip = opponent_line_length_back

                # backward is a little complicated, will try to extend my stones first
                ext_r = r
                ext_c = c
                skipped_2 = 0
                opponent_blocked = False
                for i in range(6-my_line_length_no_skip):
                    ext_r -= dr
                    ext_c -= dc
                    if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                        break
                    elif state[ext_r, ext_c] == player:
                        my_line_length_back += 1
                        opponent_blocked = True
                    elif state[ext_r, ext_c] == -player:
                        break
                    else:
                        if skipped_2 == 0:
                            skipped_2 = i + 1
                        else:
                            # peek at the next one and if it might be useful, add some interest
                            if state[ext_r-dr, ext_c-dc] == player:
                                interest_value += 15
                            break

                # see if i'm winning
                if my_line_length_back == 5:
                    # if there are 5 stones in backward counting, and it's not skipped in the middle
                    if skipped_2 == 0 or skipped_2 == (6-my_line_length_no_skip):
                        # i will win with this move, I will place the stone
                        final_single_move[0,0] = r
                        final_single_move[0,1] = c
                        return final_single_move

                # extend my forward line length to check if there is hard 4
                if skipped_2 == 0:
                    my_line_length += my_line_length_back - my_line_length_no_skip
                else:
                    my_line_length += skipped_2 - 1

                backward_my_open = True if skipped_2 > 0 else False
                backward_opponent_open = False
                # then try to extend the opponent
                if opponent_blocked == True:
                    if skipped_2 == 1:
                        backward_opponent_open = True
                    skipped_2 = 0 # reset the skipped_2 here to enable the check of opponent 5 later
                else:
                    ext_r = r
                    ext_c = c
                    skipped_2 = 0
                    for i in range(6-opponent_line_length_no_skip):
                        ext_r -= dr
                        ext_c -= dc
                        if ext_r < 0 or ext_r >= board_size or ext_c < 0 or ext_c >= board_size:
                            break
                        elif state[ext_r, ext_c] == player:
                            break
                        elif state[ext_r, ext_c] == -player:
                            opponent_line_length_back += 1
                        else:
                            if skipped_2 == 0:
                                skipped_2 = i + 1
                            else:
                                # peek at the next one and if it might be useful, add some interest
                                if state[ext_r-dr, ext_c-dc] == -player:
                                    interest_value += 15
                                break
                    # extend opponent forward line length to check if there is hard 4
                    if skipped_2 == 0:
                        opponent_line_length += opponent_line_length_back - opponent_line_length_no_skip
                    else:
                        opponent_line_length += skipped_2 - 1
                        backward_opponent_open = True
                        # here if opponent_line_length_back == 5, skipped_2 will be 0 and this flag won't be True
                        # but it do not affect our final result, because we have to block this no matter if it's open

                # check if we have to block this
                if opponent_line_length_back == 5:
                    if (skipped_2 == 0) or (skipped_2 == 6-opponent_line_length_no_skip):
                        final_single_move[0,0] = r
                        final_single_move[0,1] = c
                        force_to_block = True
                if force_to_block == False:
                    # if I will win after this move, I won't consider other moves
                    if forward_my_open == True and my_line_length == 4:
                        my_hard_4 += 1
                    if backward_my_open == True and my_line_length_back == 4:
                        my_hard_4 += 1
                    if my_hard_4 >= 2:
                        final_single_move[0,0] = r
                        final_single_move[0,1] = c
                        exist_will_win_move = True
                if force_to_block == False and exist_will_win_move == False:
                    # compute the interest_value for other moves
                    # if any line length >= 5, it's an overline so skipped
                    if (forward_my_open == True) and (my_line_length < 5):
                        interest_value += my_line_length ** 4
                    if (backward_my_open == True) and (my_line_length_back < 5):
                        interest_value += my_line_length_back ** 4
                    if (forward_opponent_open == True) and (opponent_line_length < 5):
                        interest_value += opponent_line_length ** 4
                    if (backward_opponent_open == True) and (opponent_line_length_back < 5):
                        interest_value += opponent_line_length_back ** 4
                # if (r,c) == (5,5):
                #     print("(dr,dc) =", dr,dc)
                #     print('forward_my_open', forward_my_open, "my_line_length", my_line_length)
                #     print('backward_my_open', backward_my_open,"my_line_length_back", my_line_length_back)
                #     print('forward_opponent_open',forward_opponent_open,'opponent_line_length',opponent_line_length)
                #     print('backward_opponent_open',backward_opponent_open,'opponent_line_length_back',opponent_line_length_back)
                #     print("interest_value=", interest_value)
            # after looking at all directions, record the total interest_value of this move
            move_interest_values[r, c] += interest_value
            if interest_value > 256: # one (length_4) ** 4, highly interesting move
                n_moves += 1

    # all moves have been investigated now see if we have to block first
    if force_to_block == True or exist_will_win_move == True:
        if verbose == True:
            print(final_single_move[0,0], final_single_move[0,1], "Only One")
        return final_single_move
    else:
        flattened_interest = move_interest_values.ravel()
        # The interest value > 250 means at least one length_4 or three length_3 which make it highly interesting
        #n_high_interest_moves = np.sum(flattened_interest > 266) # did it in the loop
        if n_moves > empty_spots_left:
            n_moves = empty_spots_left
        high_interest_idx = np.argsort(flattened_interest)[-n_moves:][::-1]
        interested_moves = np.empty(n_moves*2, dtype=np.int64).reshape(n_moves, 2)
        interested_moves[:,0] = high_interest_idx // board_size
        interested_moves[:,1] = high_interest_idx % board_size

        if verbose == True:
            print("There are", n_moves, "interested_moves")
            for i in range(n_moves):
                print(interested_moves[i,0],interested_moves[i,1],'  :  ', flattened_interest[high_interest_idx[i]])
        return interested_moves

@numba.jit(nopython=True, nogil=True)
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
        if line_length == 5:
            return True # 5 in a row
    return False

@numba.jit(nopython=True, nogil=True)
def i_lost(state, player):
    for r in range(board_size):
        for c in range(board_size):
            if state[r,c] == 0 and i_win(state, (r,c), -player):
                return True
    return False

@numba.jit(nopython=True, nogil=True)
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
            elif skipped_1 == 0 and state[ext_r, ext_c] == 0:
                skipped_1 = i+1 # allow one skip and record the position of the skip
            else:
                break
        # try to extend in the opposite direction
        ext_r = r
        ext_c = c
        skipped_2 = 0
        # the backward counting starts at the furthest "unskipped" stone
        if skipped_1 != 0:
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
            elif skipped_2 == 0 and state[ext_r, ext_c] == 0:
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
            if skipped_2 != 0:
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
        if line_length == 4 and skipped_1 != 0:
            n_hard_4 += 1 # forward hard 4
            if n_hard_4 == 2:
                return True # two hard 4 or free 4
    return False



from collections import OrderedDict
class LRU(OrderedDict):
    'Limit size, evicting the least recently looked-up key when full'

    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        super().__init__()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

class LeveledCache:
    """
    Cache with level system, level 0 has highest priority
    When maxsize reached, oldest key from lowest cache will be deleted
    """

    def __init__(self, maxlevel, maxsize=128):
        assert maxlevel >= 0
        self.maxlevel = maxlevel
        self.maxsize = maxsize
        self.caches = [OrderedDict() for _ in range(maxlevel+1)]
        self.size = 0
    
    def get(self, key, max_accepted_level):
        """
        Go over each level in cache, find one value with key
        If none found, return None
        """
        for level in range(min(self.maxlevel, max_accepted_level)+1):
            # starting from level 0, look for cached value
            cache = self.caches[level]
            try:
                result = cache[key]
                cache.move_to_end(key)
                return result
            except KeyError:
                pass
        return None
        
    def set(self, key, value, level):
        """
        set a value in cache with level
        """
        assert level <= self.maxlevel
        # delete oldest from lowest cache if size reached
        if self.size >= self.maxsize:
            for l in range(self.maxlevel, -1, -1):
                cache = self.caches[l]
                try:
                    oldest = next(iter(cache))
                    del cache[oldest]
                    break
                except StopIteration:
                    # if cache is empty skip and go to the next level cache
                    pass
        else:
            # update size
            self.size += 1
        # insert new key
        cache = self.caches[level]
        # move key to end
        if key in cache:
            cache.move_to_end(key)
        # set the value
        cache[key] = value



def read_board_state(f):
    # default
    black_stones = []
    white_stones = []
    board = [black_stones, white_stones]
    last_move = None
    playing = 0
    # read and parse board
    for line in open(f):
        if '|' in line:
            line_idx, contents = line.split('|', maxsplit=1)
            row_i = int(line_idx)
            stones = contents.split()
            if len(stones) == board_size:
                for col_j, s in enumerate(stones):
                    if s == 'x':
                        black_stones.append((row_i, col_j))
                    elif s == 'X':
                        black_stones.append((row_i, col_j))
                        last_move = (row_i, col_j)
                        playing = 0
                    elif s == 'o':
                        white_stones.append((row_i, col_j))
                    elif s == 'O':
                        white_stones.append((row_i, col_j))
                        last_move = (row_i, col_j)
                        playing = 1
                    elif s == '-':
                        pass
                    else:
                        print(f'found unknown stone: {s}')
    board_state = [board, last_move, playing, board_size]
    return board_state

def convert_board_state(board_state):
    board, last_move, playing, board_size = board_state
    playing_white = bool(playing)
    # build new state representation
    state = np.zeros(board_size**2, dtype=np.int8).reshape(board_size, board_size)
    # put black stones, update index
    for br, bc in board[0]:
        state[br-1,bc-1] = 1
    # put white stones, update index
    for wr, wc in board[1]:
        state[wr-1,wc-1] = -1
    return state

def show_state(state):
    board_size = 15
    print(' '*4 + ' '.join([chr(97+i) for i in range(board_size)]))
    print(' '*3 + '='*(2*board_size))
    for x in range(board_size):
        row = ['%2s|'%x]
        for y in range(board_size):
            if state[x,y] == 1:
                c = 'x'
            elif state[x,y] == -1:
                c = 'o'
            else:
                c = '-'
            row.append(c)
        print(' '.join(row))

def benchmark():

    state = np.zeros(board_size**2, dtype=np.int8).reshape(board_size, board_size)
    state[8,8] = 1
    move_interest_values = np.zeros(board_size**2, dtype=np.float32).reshape(board_size,board_size)
    player = 1
    n_moves = 20
    # find_interesting_moves(state, 224, move_interest_values, player, n_moves, verbose=False)

    # find_interesting_moves_gpu(state, 224, move_interest_values, player, n_moves, verbose=False)
    t0 = time.time()
    N = 1000
    for _ in range(N):
        find_interesting_moves(state, 224, move_interest_values, player, n_moves, verbose=False)

        # find_interesting_moves_gpu(state, 224, move_interest_values, player, n_moves, verbose=False)
    print(f"{N} repeats took {time.time() - t0:9.3f} s")


if __name__ == "__main__":
    benchmark()