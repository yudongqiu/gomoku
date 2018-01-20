#!/usr/bin/env python
# -- coding: utf-8 --

#==========================
#=      Gomoku Game       =
#==========================

from __future__ import print_function, division
import os, sys, time, collections, shutil
from functools import update_wrapper
import pickle, h5py
import numpy as np
from nifty import createWorkQueue,getWorkQueue,queue_up,wq_wait,LinkFile
import tarfile

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
                action = current_player.strategy(self.state)
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
            for i in range(1,4):
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
            if len(winning_stones) >= 5:
                self.winning_stones = winning_stones
                return self.players[self.playing].name
        return None

    def delay(self, n):
        """ Delay n seconds if not in fastmode"""
        if not self.fastmode:
            time.sleep(n)

    def get_strategy(self, p):
        return p.strategy(self.state)

class Player(object):
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        try:
            self._name = str(value)
        except:
            raise TypeError("Player Name must be a string.")

    def __repr__(self):
        return "Player %s"%self.name

    def __init__(self, name):
        self.name = name
        # search for the strategy file
        # p = __import__(name)
        # p.initialize()
        # self.strategy = p.strategy
        # self.finish = p.finish
        # self.train_model = p.train_model

def prepare_train_data(learndata_A, learndata_B):
    nb, nw = len(learndata_A), len(learndata_B)
    n_data = nb + nw
    train_X = np.empty(n_data*15*15*3, dtype=np.int8).reshape(n_data,15,15,3)
    train_Y = np.empty(n_data, dtype=np.float32).reshape(-1,1)
    i = 0
    bx, wx, by, wy = [],[],[],[]
    for k in learndata_A:
        x, y, n = learndata_A[k]
        bx.append(x)
        by.append(y)
        train_X[i, :, :, 0] = (x == 1) # first plane indicates my stones
        train_X[i, :, :, 1] = (x == -1) # first plane indicates opponent_stones
        train_X[i, :, :, 2] = 1 # third plane indicates if i'm black
        train_Y[i, 0] = y
        i += 1
    for k in learndata_B:
        x, y, n = learndata_B[k]
        wx.append(x)
        wy.append(y)
        train_X[i, :, :, 0] = (x == 1) # first plane indicates my stones
        train_X[i, :, :, 1] = (x == -1) # first plane indicates opponent_stones
        train_X[i, :, :, 2] = 0 # third plane indicates if i'm black
        train_Y[i, 0] = y
        i += 1
    # save the current train data to data.h5 file
    h5f = h5py.File('data.h5','w')
    h5f.create_dataset('bx',data=np.array(bx, dtype=np.int8))
    h5f.create_dataset('by',data=np.array(by, dtype=np.float32))
    h5f.create_dataset('wx',data=np.array(wx, dtype=np.int8))
    h5f.create_dataset('wy',data=np.array(wy, dtype=np.float32))
    h5f.close()
    print("Successfully prepared %d black and %d white training data." % (nb, nw))
    return train_X, train_Y

def update_learn_data(learndata1, learndata2):
    # update learndata1 with data in learndata2, if n2 > n1 then use y2
    for key in learndata2:
        x2, y2, n2 = learndata2[key]
        if key in learndata1:
            x1, y1, n1 = learndata1[key]
            if n2 > n1:
                 learndata1[key] = x2, y2, n2
        else:
            learndata1[key] = (x2, y2, n2)

def main():
    import argparse
    parser = argparse.ArgumentParser("Play the Gomoku Game!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--n_train', type=int, default=10, help='Play a number of games to gather statistics.')
    parser.add_argument('-b', '--n_batches', type=int, default=10, help='Number of batches of games to play in each iteration.')
    parser.add_argument('-w', '--n_workers', type=int, default=30, help='Number of workers to use in each batch.')
    parser.add_argument('-g', '--worker_games', type=int, default=100, help='Number of games each worker play each time before returning data.')
    parser.add_argument('-p', '--wq_port', type=int, default=50123, help='Port to use in work queue.')
    args = parser.parse_args()



    createWorkQueue(args.wq_port, name='gtrain')
    wq = getWorkQueue()
    cmdstr = 'python gomoku_worker.py -g %d' % args.worker_games
    input_files = ['construct_dnn.py', 'player_A.py', 'player_B.py']

    print("Training the model for %d iterations, each will run %d batches of games."% (args.n_train, args.n_batches))


    model_name = 'initial_model' # we start with tf_model saved in initial_model
    import construct_dnn
    model = construct_dnn.construct_dnn()
    model.load(os.path.join(model_name, 'tf_model', 'tf_model'))

    for i_train in xrange(args.n_train):
        prev_model_name = model_name
        # check if the current model exists
        model_name = "trained_model_%03d" % i_train
        if os.path.exists(model_name):
            backup_name = model_name+'_backup'
            if os.path.exists(backup_name):
                shutil.rmtree(backup_name)
            shutil.move(model_name, backup_name)
            print("Current model %s already exists, backed up to %s" % (model_name, backup_name))
        # create black_learndata and white_learndata dict
        black_learndata, white_learndata = dict(), dict()
        # create and enter the model folder
        os.mkdir(model_name)
        os.chdir(model_name)
        for i_batch in xrange(args.n_batches):
            print("Batch %d, launching %d workers, each play %d games, then update strategy.learndata" % (i_batch, args.n_workers, args.worker_games))
            batch_name = "batch_%03d" % i_batch
            os.mkdir(batch_name)
            os.chdir(batch_name)
            pickle.dump(black_learndata, open('black.learndata', 'wb'))
            pickle.dump(white_learndata, open('white.learndata', 'wb'))
            # put all input files in a tar.gz file for transfer
            with tarfile.open("input.tar.gz", "w:gz") as tar:
                for f in input_files:
                    tar.add("../../" + f, arcname=f)
                tar.add('black.learndata')
                tar.add('white.learndata')
                # add the previous tf model to the input files, rename to tf_model
                tar.add(os.path.join('../..', prev_model_name, 'tf_model'), arcname='tf_model')
            for i_worker in xrange(args.n_workers):
                worker_name = "worker_%03d" % i_worker
                os.mkdir(worker_name)
                os.chdir(worker_name)
                LinkFile('../input.tar.gz', 'input.tar.gz')
                LinkFile('../../../gomoku_worker.py', 'gomoku_worker.py')
                queue_up(wq, command = cmdstr, input_files = ['input.tar.gz', 'gomoku_worker.py'], output_files = ['output.tar.gz'])
                os.chdir('..')
            # after all workers in this batch finish, collect and update strategy.learndata
            wq_wait(wq)
            print("All workers finished! Extracting and updating learndata files.")
            black_learndata = dict()
            white_learndata = dict()
            for i_worker in xrange(args.n_workers):
                worker_name = "worker_%03d" % i_worker
                os.chdir(worker_name)
                with tarfile.open("output.tar.gz") as tar:
                    tar.extractall()
                newblack_learndata = pickle.load(open('newblack.learndata'))
                print("%d new black learndata loaded from %s" % (len(newblack_learndata), worker_name))
                update_learn_data(black_learndata, newblack_learndata)
                print("black.learndata updated to %d data" % len(black_learndata))
                newwhite_learndata = pickle.load(open('newwhite.learndata'))
                print("%d new white learndata loaded from %s" % (len(newwhite_learndata), worker_name))
                update_learn_data(white_learndata, newwhite_learndata)
                print("white.learndata updated to %d data" % len(white_learndata))
                os.chdir('..')
            os.chdir('..')
        # when all batches of games finished, the final learndata should be used to train a model
        train_X, train_Y = prepare_train_data(black_learndata, white_learndata)
        # fit the tf model
        model.fit(train_X, train_Y, n_epoch=10, validation_set=0.1, show_metric=True)
        os.mkdir('tf_model')
        model.save('tf_model/tf_model')
        print("\n ---===   Model %s saved!   ===---" % model_name)
        # finished current model, goto next iteration
        os.chdir("..")

if __name__ == "__main__":
    main()
