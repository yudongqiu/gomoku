#!/usr/bin/env python
# -- coding: utf-8 --

#==========================
#=      Gomoku Game       =
#==========================

from __future__ import print_function, division
import os, sys, time, collections
from functools import update_wrapper
import pickle

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
        self.players = [Player(player_name) for player_name in players]
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
        p = __import__(name)
        p.initialize()
        self.strategy = p.strategy
        self.finish = p.finish


def main():
    import argparse
    parser = argparse.ArgumentParser("Play the Gomoku Game!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--first_center', action='store_true', help='The first move must be on the center of board?')
    parser.add_argument('--fast', action='store_true', help='Run the game in fast mode.')
    parser.add_argument('-n', '--ngames', type=int, help='Play a number of games to gather statistics.')
    parser.add_argument('--load', help='Load a previously saved state to continue the game.')
    args = parser.parse_args()

    game = Gomoku(board_size=15, players=['model2', 'model1'], fastmode=args.fast, first_center=args.first_center)
    if args.load:
        state = pickle.load(open(args.load,'rb'))
        game.load_state(state)

    if args.ngames is None:
        game.play()
    else:
        # check if all players have stategy function setup
        for p in game.players:
            if p.strategy.__name__ == 'human_input':
                print("%s need a strategy function to enter the auto-play mode. Exiting.."%p.name)
                return
        print("Gathering result of %d games..."%args.ngames)
        game.fastmode = 2
        game_output = open('game_results.txt','w')
        winner_board = collections.OrderedDict([(p.name, 0) for p in game.players])
        winner_board["Draw"] = 0
        def playone(i):
            game_output.write('Game %-4d .'%(i+1))
            game.reset()
            winner = game.play()
            winner_board[winner] += 1
            game_output.write('Game %-4d: Winner is %s\n'%(i+1, winner))
            game_output.flush()
        for i in range(args.ngames):
            playone(i)
        game_output.close()
        print("Name    |   Games Won")
        for name, nwin in winner_board.items():
            print("%-7s | %7d"%(name, nwin))
    # Let the players finish their game
    for p in game.players:
        if hasattr(p, 'finish'):
            p.finish()

if __name__ == "__main__":
    main()
