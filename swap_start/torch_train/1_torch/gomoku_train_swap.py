#!/usr/bin/env python
# -- coding: utf-8 --

#==========================
#=      Gomoku Game       =
#==========================

import os
import shutil
import copy
import sys
import time
import collections
import pickle
import h5py
import numpy as np
import random
from functools import update_wrapper

def decorator(d):
    "Make function d a decorator: d wraps a function fn."
    def _d(fn):
        return update_wrapper(d(fn), fn)
    update_wrapper(_d, d)
    return _d

@decorator
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
            return f(args)
    _f.cache = cache
    return _f

@memo
def colored(s, color=''):
    if color.lower() == 'green':
        return '\033[92m' + s + '\033[0m'
    elif color.lower() == 'yellow':
        return '\033[93m' + s + '\033[0m'
    elif color.lower() == 'red':
        return '\033[91m' + s + '\033[0m'
    elif color.lower() == 'blue':
        return '\033[94m' + s + '\033[0m'
    elif color.lower() == 'bold':
        return '\033[1m' + s + '\033[0m'
    else:
        return s

class Gomoku(object):
    """ Gomoku Game Rules:
    Two players alternatively put their stone on the board. First one got five in a row wins.
    """

    def __init__(self, board_size=15, players=None, fastmode=False, first_center=None):
        print("*********************************")
        print("*      Welcome to Gomoku !      *")
        print("*********************************")
        print(self.__doc__)
        self.reset()
        self.board_size = board_size
        self.fastmode = fastmode
        #self.players = [Player(player_name) for player_name in players]
        self.first_center = first_center

    @property
    def state(self):
        return (self.board, self.last_move, self.playing, self.board_size)

    def load_state(self, state):
        (self.board, self.last_move, self.playing, self.board_size) = state

    def reset(self):
        self.board = (set(), set())
        self.playing = None
        self.winning_stones = set()
        self.last_move = None

    def print_board(self):
        print(' '*4 + ' '.join([chr(97+i) for i in range(self.board_size)]))
        print(' '*3 + '='*(2*self.board_size))
        for x in range(1, self.board_size+1):
            row = ['%2s|'%x]
            for y in range(1, self.board_size+1):
                if (x,y) in self.board[0]:
                    c = 'x'
                elif (x,y) in self.board[1]:
                    c = 'o'
                else:
                    c = '-'
                if (x,y) in self.winning_stones or (x,y) == self.last_move:
                    c = colored(c, 'green')
                row.append(c)
            print(' '.join(row))

    def play(self):
        if self.fastmode < 2:  print("Game Start!")
        i_turn = len(self.board[0]) + len(self.board[1])
        new_step = None
        while True:
            if self.fastmode < 2:  print("----- Turn %d -------" % i_turn)
            self.playing = i_turn % 2
            if self.fastmode < 2:
                self.print_board()
            current_player = self.players[self.playing]
            other_player = self.players[int(not self.playing)]
            if self.fastmode < 2: print("--- %s's turn ---" % current_player.name)
            max_try = 5
            for i_try in range(max_try):
                action, _ = current_player.strategy(self.state)
                if action == (0, 0):
                    print("Player %s admit defeat!" % current_player.name)
                    winner = other_player.name
                    self.print_board()
                    print("Winner is %s"%winner)
                    return winner
                self.last_move = action
                if self.place_stone() is True:
                    break
                if i_try == max_try-1:
                    print("Player %s has made %d illegal moves, he lost."%(current_player.name, max_try))
                    winner = other_player.name
                    print("Winner is %s"%winner)
                    return winner
            # check if current player wins
            winner = self.check_winner()
            if winner:
                self.print_board()
                print("##########    %s is the WINNER!    #########" % current_player.name)
                return winner
            elif i_turn == self.board_size ** 2 - 1:
                self.print_board()
                print("This game is a Draw!")
                return "Draw"
            i_turn += 1

    def place_stone(self):
        # check if this position is on the board
        r, c = self.last_move
        if r < 1 or r > self.board_size or c < 1 or c > self.board_size:
            print("This position is outside the board!")
            return False
        # check if this position is already taken
        taken_pos = self.board[0] | self.board[1]
        if self.first_center is True and len(taken_pos) == 0:
            # if this is the very first move, it must be on the center
            center = int((self.board_size+1)/2)
            if r != center or c != center:
                print("This is the first move, please put it on the center (%s%s)!"% (str(center),chr(center+96)))
                return False
        elif self.last_move in taken_pos:
            print("This position is already taken!")
            return False
        self.board[self.playing].add(self.last_move)
        return True

    def check_winner(self):
        r, c = self.last_move
        my_stones = self.board[self.playing]
        # find any nearby stone
        nearby_stones = set()
        for x in range(max(r-1, 1), min(r+2, self.board_size+1)):
            for y in range(max(c-1, 1), min(c+2, self.board_size+1)):
                stone = (x,y)
                if stone in my_stones and (2*r-x, 2*c-y) not in nearby_stones:
                    nearby_stones.add(stone)
        for nearby_s in nearby_stones:
            winning_stones = {self.last_move, nearby_s}
            nr, nc = nearby_s
            dx, dy = nr-r, nc-c
            # try to extend in this direction
            for i in range(1,5):
                ext_stone = (nr+dx*i, nc+dy*i)
                if ext_stone in my_stones:
                    winning_stones.add(ext_stone)
                else:
                    break
            # try to extend in the opposite direction
            for i in range(1,5):
                ext_stone = (r-dx*i, c-dy*i)
                if ext_stone in my_stones:
                    winning_stones.add(ext_stone)
                else:
                    break
            if len(winning_stones) == 5:
                self.winning_stones = winning_stones
                return self.players[self.playing].name
        return None

    def delay(self, n):
        """ Delay n seconds if not in fastmode"""
        if not self.fastmode:
            time.sleep(n)

    def get_strategy(self, p):
        return p.strategy(self.state)

def prepare_train_data(learndata_A, learndata_B):
    nb, nw = len(learndata_A), len(learndata_B)
    n_data = nb + nw
    train_X = np.empty(n_data*15*15*3, dtype=np.int8).reshape(n_data,3,15,15)
    train_Y = np.empty(n_data, dtype=np.float32).reshape(-1,1)
    train_W = np.empty(n_data, dtype=np.float32)
    i = 0
    bx, wx, by, wy = [],[],[],[]
    for k in list(learndata_A.keys()):
        x, y, n = learndata_A.pop(k)
        # import IPython; IPython.embed(); return;
        bx.append(x)
        by.append(y)
        train_X[i, 0, :, :] = (x == 1) # first plane indicates my stones
        train_X[i, 1, :, :] = (x == -1) # first plane indicates others stones
        train_X[i, 2, :, :] = 1 # third plane indicates if i'm black
        train_Y[i, 0] = y
        train_W[i] = n ** 0.5
        i += 1
    for k in list(learndata_B.keys()):
        x, y, n = learndata_B.pop(k)
        wx.append(x)
        wy.append(y)
        train_X[i, 0, :, :] = (x == -1) # first plane indicates my stones, which is white
        train_X[i, 1, :, :] = (x == 1) # first plane indicates others stones, which is black
        train_X[i, 2, :, :] = 0 # third plane indicates if i'm black
        train_Y[i, 0] = y
        train_W[i] = n ** 0.5
        i += 1
    # save the current train data to data.h5 file
    h5f = h5py.File('data.h5','w')
    h5f.create_dataset('bx',data=np.array(bx, dtype=np.int8))
    h5f.create_dataset('by',data=np.array(by, dtype=np.float32))
    h5f.create_dataset('wx',data=np.array(wx, dtype=np.int8))
    h5f.create_dataset('wy',data=np.array(wy, dtype=np.float32))
    h5f.create_dataset('w',data=train_W)
    h5f.close()
    print("Successfully prepared %d black and %d white training data." % (nb, nw))
    return train_X.astype(np.float32), train_Y, train_W

def load_data_h5(fnm):
    h5f = h5py.File(fnm,'r')
    bx = h5f['bx']
    by = h5f['by']
    wx = h5f['wx']
    wy = h5f['wy']
    traint_W = h5f['w'][:]
    nb, nw = len(bx), len(wx)
    n_data = nb + nw
    train_X = np.empty(n_data*15*15*3, dtype=np.int8).reshape(n_data,3,15,15)
    train_Y = np.empty(n_data, dtype=np.float32).reshape(-1,1)
    # fill in the data for black
    train_X[:nb, 0, :, :] = np.equal(bx, 1)
    train_X[:nb, 1, :, :] = np.equal(bx, -1)
    train_X[:nb, 2, :, :] = 1
    train_Y[:nb, 0] = by
    # fill in the data for white
    train_X[nb:, 0, :, :] = np.equal(wx, -1)
    train_X[nb:, 1, :, :] = np.equal(wx, 1)
    train_X[nb:, 2, :, :] = 0
    train_Y[nb:, 0] = wy
    h5f.close()
    return train_X.astype(np.float32), train_Y, traint_W


def gen_begin_board(allstones, begin_lib=None):
    if begin_lib == None:
        stones = random.sample(allstones, 3)
    else:
        stones = random.choice(begin_lib)
    board = (set(), set())
    for i,stone in enumerate(stones):
        board[i%2].add(stone)
    return board

MODEL_FILE = 'dnn_model.pt'

def main():
    import argparse
    parser = argparse.ArgumentParser("Play the Gomoku Game!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--n_train', type=int, default=10, help='Play a number of games to gather statistics.')
    parser.add_argument('-t', '--train_step', type=int, default=100, help='Train a new model after this number of games.')
    parser.add_argument('-e', '--n_epoch', type=int, default=100, help="Number of epochs for each training model")
    parser.add_argument('-l', '--begin_lib', help='Begin board library file')
    parser.add_argument('-p', '--begin_lib_p', type=float, default=1.0, help='Possibility of begin lib to be used')
    parser.add_argument('-b', '--benchmark', action='store_true', default=False, help='Enable benchmark after each training model')
    args = parser.parse_args()

    game = Gomoku(board_size=15, first_center=False)

    # load model
    from dnn_model import get_new_model, load_existing_model, save_model
    model = get_new_model()
    # search for existing trained model
    last_i_train = -1
    for i_train in range(args.n_train):
        # check if the current model exists
        model_name = "trained_model_%03d" % i_train
        if os.path.exists(model_name):
            print(f"Folder {model_name} exists")
            last_i_train = i_train
        else:
            break
    if last_i_train >= 0:
        # try to load the last trained model
        model_name = "trained_model_%03d" % last_i_train
        model_fnm = os.path.join(model_name, MODEL_FILE)
        if os.path.exists(model_fnm):
            model = load_existing_model(model_fnm)
            print(f"Loaded trained model from {model_fnm}")
        else:
            # if last trained model not exist, load the previous model
            if last_i_train > 0:
                prev_model_name = f"trained_model_{last_i_train-1:03d}"
                prev_model_fnm = os.path.join(prev_model_name, MODEL_FILE)
                model = load_existing_model(prev_model_fnm)
                print(f"Loaded lastest model from {prev_model_fnm}")
            # try to reuse data and start training
            train_data_fnm = os.path.join(model_name, 'data.h5')
            if os.path.exists(train_data_fnm):
                train_X, train_Y, train_W = load_data_h5(train_data_fnm)
                print(f"Training data loaded from {train_data_fnm}, start training")
                model.fit(train_X, train_Y, epochs=args.n_epoch, validation_split=0.2)
                save_model(model, model_fnm)
                print("Model %s saved!" % model_name)
            else:
                # delete this folder and start again
                shutil.rmtree(model_name)
                print(f"Deleting folder {model_name}")
                last_i_train -= 1


    from AIPlayer import AIPlayer
    player_A = AIPlayer('Black', model)
    player_B = AIPlayer('White', model)
    # set up linked learndata and cache (allow AI to look into opponent's data)
    player_A.opponent = player_B
    player_B.opponent = player_A

    game.players = [player_A, player_B]
    if args.train_step > 1:
        game.fastmode = 2
    else:
        player_A.show_q = player_B.show_q = True

    allstones = set([(r,c) for r in range(1,16) for c in range(1,16)])
    if args.begin_lib != None:
        begin_lib = __import__(args.begin_lib).begin_lib
    else:
        begin_lib = None

    def playone(i, game_output, winner_board, replay=False):
        game.reset()
        player_A.reset()
        player_B.reset()
        if replay:
            game.board = copy.deepcopy(game.last_begin_board)
        else:
            if random.random() < args.begin_lib_p:
                game.board = gen_begin_board(allstones, begin_lib)
            else:
                game.board = gen_begin_board(allstones, None)
            # store the begin board
            game.last_begin_board = copy.deepcopy(game.board)
        # randomly assign a black stone to be the last move
        game.last_move = next(iter(game.board[0]))
        winner = game.play()
        winner_board[winner] += 1
        game_output.write('Game %-4d: Winner is %s\n'%(i+1, winner))
        game_output.flush()

    print("Training the model for %d iterations."%args.n_train)

    for i_train in range(last_i_train+1, args.n_train):
        model_name = "trained_model_%03d" % i_train
        # create and enter the model folder
        os.mkdir(model_name)
        os.chdir(model_name)
        # play the games
        print("Training model %s" % model_name)
        winner_board = dict([(p.name, 0) for p in game.players])
        winner_board['Draw'] = 0
        with open('game_results.txt','w') as game_output:
            replay_last_game = False
            i_game = 0
            repeating_n = 0
            repeat_n_after_surprise = 0
            while True:
                playone(i_game, game_output, winner_board, replay=replay_last_game)
                surprised = False # any(player.surprised for player in game.players)
                replay_last_game = False
                # if surprised:
                #     replay_last_game = True
                #     repeat_n_after_surprise = 0
                # elif repeat_n_after_surprise < 5:
                #     # keep replaying at least 5 games after surprise
                #     replay_last_game = True
                n_used = len(player_A.learndata)
                max_data_count = 3000000
                if replay_last_game:
                    repeating_n += 1
                    repeat_n_after_surprise += 1
                    print(f"Game {i_game} repeating {repeating_n} | {repeat_n_after_surprise}: {game.last_begin_board} | data {n_used//1000}k/{max_data_count//1000}k")
                else:
                    repeating_n = 0
                    repeat_n_after_surprise = 0
                    i_game += 1
                    if i_game >= args.train_step:
                        break
                    print(f"New game {i_game}: {format_begin_board(game.last_begin_board)} | data {n_used//1000}k/{max_data_count//1000}k")
                # prevent memory overflow and getting killed
                if n_used > max_data_count:
                    print('Learn data is full, stopping')
                    break

        print("Name    |   Games Won")
        for name, nwin in winner_board.items():
            print("%-7s | %7d"%(name, nwin))
        # reset player cache
        player_A.reset_cache()
        player_B.reset_cache()
        # share the cache
        player_B.cache = player_A.cache
        # collect training data
        train_X, train_Y, train_W = prepare_train_data(player_A.learndata, player_B.learndata)
        # fit the model
        # import IPython; IPython.embed()
        model.fit(train_X, train_Y, epochs=args.n_epoch, validation_split=0.2)
        save_model(model, MODEL_FILE)
        print("Model %s saved!" % model_name)
        os.chdir('..')
        if args.benchmark and i_train > 0:
            prev_model_name = f"trained_model_{i_train-1:03d}"
            prev_model_name = os.path.join(prev_model_name, MODEL_FILE)
            prev_model = load_existing_model(prev_model_name)
            os.chdir(model_name)
            with open('benchmark.txt','w') as game_output:
                # play 1000 games as Black
                print("New model as Black", file=game_output)
                player_A.model, player_B.model = model, prev_model
                winner_board_b = dict([(p.name, 0) for p in game.players])
                winner_board_b['Draw'] = 0
                for i_game in range(100):
                    playone(i_game, game_output, winner_board_b)
                print("Name    |   Games Won", file=game_output)
                for name, nwin in winner_board_b.items():
                    print("%-7s | %7d"%(name, nwin), file=game_output)
                # play 100 games as White
                print("New model as White", file=game_output)
                player_A.model, player_B.model = prev_model, model
                winner_board_w = dict([(p.name, 0) for p in game.players])
                winner_board_w['Draw'] = 0
                for i_game in range(100):
                    playone(i_game, game_output, winner_board_w)
                print("Name    |   Games Won", file=game_output)
                for name, nwin in winner_board_w.items():
                    print("%-7s | %7d"%(name, nwin), file=game_output)
                print('\n\n'+'-' * 50, file=game_output)
                print(f'           |   Win           Lose           Draw', file=game_output)
                print('-' * 50, file=game_output)
                print(f' as Black  | {winner_board_b[player_A.name]:10} {winner_board_b[player_B.name]:10} {winner_board_b["Draw"]:10}', file=game_output)
                print(f' as White  | {winner_board_w[player_B.name]:10} {winner_board_w[player_A.name]:10} {winner_board_w["Draw"]:10}', file=game_output)
                print('-' * 50, file=game_output)
            os.chdir('..')
            # refresh the training by loading it back
            model_fnm = os.path.join(model_name, MODEL_FILE)
            model = load_existing_model(model_fnm)
            player_A.model = player_B.model = model

def format_begin_board(begin_board):
    black_stones, white_stones = begin_board
    bs = ','.join(f'({r:2},{c:2})' for r,c in black_stones)
    ws = ','.join(f'({r:2},{c:2})' for r,c in white_stones)
    return f"{bs} | {ws}"

def get_state(x1):
    is_black = bool(x1[2,0,0])
    return (x1[0] - x1[1]) if is_black else (x1[1] - x1[0])

if __name__ == "__main__":
    main()
