#!/usr/bin/env python

from __future__ import print_function, division
import itertools, time, copy
import collections, random
import os, pickle
import numba
import numpy as np

board_size = 15
show_q = False

class AIPlayer:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.learndata = dict()
        self.opponent = None
        self.all_interest_states = np.zeros(board_size**4 * 3, dtype=np.float16).reshape(board_size**2, board_size, board_size, 3)
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
        self.tf_cache = LRU(maxsize=1000000)

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
        my_stones = board[playing]
        opponent_stones = board[1-playing]
        last_move = (last_move[0]-1, last_move[1]-1)
        # build new state representation
        state = np.zeros(board_size**2, dtype=np.int8).reshape(board_size, board_size)
        for i,j in my_stones:
            state[i-1,j-1] = 1
        for i,j in opponent_stones:
            state[i-1,j-1] = -1
        # prepare input for best_action_q
        alpha = -2.0
        beta = 2.0
        empty_spots_left = board_size**2 - len(my_stones) - len(opponent_stones) # np.sum(state==0)
        # predict next best action and q
        best_move, best_q = self.best_action_q(state, empty_spots_left, alpha, beta, 1)
        # update winrate if game finish
        self.update_if_game_finish(state, best_move, best_q, empty_spots_left)
        # return the best move
        return (best_move[0]+1, best_move[1]+1)

    def best_action_q(self, state, empty_spots_left, alpha, beta, player):
        """ get the optimal action for a state and the predicted win rate 
        Params
        ------
        state: np.ndarray of shape (15, 15)
            The current game state in a matrix. 1 is my stone, -1 is opponent stone, 0 is empty
        empty_spots_left: int
            How many empty spots are left, easy to keep track
        alpha: float
            Current alpha value in alpha-beta pruning, the running min of the max win rate
        beta: float
            Current beta value in alpha-beta pruning, the running max of the min win rate
        player: int
            The current player. 1 is me, -1 is opponent

        Returns
        -------
        best_move: tuple(int, int)
            The best move on the board, given by (r, c)
        best_q: float
            The value the best move. 1.0 means 100% win, -1.0 means 100% lose, 0 means draw
        """
        if empty_spots_left == 0: # Board filled up, it's a tie
            return None, 0.0
        verbose = False
        n_moves = 40 if empty_spots_left > 200 else 20
        self.move_interest_values.fill(0) # reuse the same array to save init cost
        self.move_interest_values[4:11, 4:11] = 5.0 # manually assign higher interest in middle
        interested_moves = find_interesting_moves(state, empty_spots_left, self.move_interest_values, player, n_moves, verbose)
        #best_move = (-1,-1) # admit defeat if all moves have 0 win rate
        best_move = (interested_moves[0,0], interested_moves[0,1]) # continue to play even I'm losing
        if len(interested_moves) == 1:
            state[best_move] = player # temporarily put the stone down
            if i_lost(state, player):
                # if i lost after putting this stone, return -1.0 win rate
                best_q = -1.0 if player == 1 else 1.0
                return best_move, best_q
            if i_will_win(state, best_move, player):
                # if i will win no matter what opponent does, return 1.0 win rate
                best_q = 1.0 if player == 1 else -1.0
                return best_move, best_q
            state[best_move] = 0 # reset the temporarily stone
        # find the known moves among interested_moves
        tf_moves = [] # all unknown moves will be evaluated by tf_evaluate_max_u
        tf_move_ids = []
        max_q = -1.0
        for this_move in interested_moves:
            this_move = (this_move[0], this_move[1])
            assert state[this_move] == 0 # interest move should be empty here
            state[this_move] = 1
            this_state_id = state.tobytes()
            state[this_move] = 0
            cached_q = None
            # try read from learndata
            try:
                cached_q = self.learndata[this_state_id][1]
            except KeyError:
                pass
            # try use cache
            if cached_q is None:
                try:
                    cached_q = self.tf_cache[this_state_id]
                except KeyError:
                    pass
            # add to compute list
            if cached_q is not None:
                if cached_q > max_q:
                    max_q = cached_q
                    best_move = this_move
            else:
                tf_moves.append(this_move)
                tf_move_ids.append(this_state_id)
        # n_found = len(interested_moves) - len(tf_moves)
        # if n_found > 0:
        #     print(f'Found {n_found} moves in learndata')
        # run tensorflow model predict
        n_tf = len(tf_moves)
        if n_tf > 0:
            all_interest_states = self.all_interest_states[:n_tf] # we only need a slice of the big array
            all_interest_states[:,:,:,0] = (state == 1)
            all_interest_states[:,:,:,1] = (state == -1)
            all_interest_states[:,:,:,2] = 0 if self.playing_white else 1 # if I'm black, next is me so black
            for i,current_move in enumerate(tf_moves):
                ci, cj = current_move
                all_interest_states[i,ci,cj,0] = 1 # put current move down
            predict_y = self.model.predict(all_interest_states, batch_size=n_tf)
            predict_y = np.array(predict_y).flatten()
            # store predict result in cache
            for move_id, y in zip(tf_move_ids, predict_y):
                self.tf_cache[move_id] = y
            # find the largest y
            idx = np.argmax(predict_y)
            if predict_y[idx] > max_q:
                max_q = predict_y[idx]
                best_move = tf_moves[idx]
        return best_move, max_q

    def update_if_game_finish(self, state, best_move, best_q, empty_spots_left):
        # store data for this step in opponent learn data
        opponent_state = -state
        opponent_state_id = opponent_state.tobytes()
        opponent_q = -best_q
        if opponent_state_id not in self.opponent.learndata:
            self.opponent.learndata[opponent_state_id] = [opponent_state, opponent_q, 1]
        # record the history states
        self.hist_states.append(opponent_state_id)
        # check if game finish
        state[best_move] = 1
        game_result = None
        new_u = 0
        if i_win(state, best_move, 1):
            new_u = -1.0
            game_result = 'win'
        elif i_lost(state, 1):
            new_u = 1.0
            game_result = 'lose'
        elif empty_spots_left <= 2:
            new_u = 0
            game_result = 'draw'
        if game_result and self.started_from_beginning is True:
            discount = 0.9
            for opponent_state_id in self.hist_states[::-1]:
                st, u, n_visited = self.opponent.learndata[opponent_state_id]
                n_visited += 1
                new_u = u + discount * (new_u - u) / n_visited**0.5 # this is the learning rate
                # surprise
                if (game_result == 'win' and new_u > 0.1) or (game_result == 'lose' and new_u < -0.1):
                    self.surprised = True
                self.opponent.learndata[opponent_state_id] = (st, new_u, n_visited)
                print(f"Updated U from {u:9.6f} to {new_u:9.6f} [{n_visited}]")
            print(f"{self.name}: Updated win rate of {len(self.hist_states)} states")
            self.started_from_beginning = False # we only update once            




















# Below are utility functions

@numba.jit(nopython=True, nogil=True)
def find_interesting_moves(state, empty_spots_left, move_interest_values, player, n_moves, verbose=False):
    """ Look at state and find the interesing n_move moves.
    input:
    -------
    state: numpy.array board_size x board_size
    empty_spots_left: number of empty spots on the board
    player: 1 or -1, the current player
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
        if line_length == 5:
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