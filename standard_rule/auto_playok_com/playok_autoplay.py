#!/usr/bin/env python

from __future__ import division
import numpy as np
import pyautogui
from Xlib import display, X
from PIL import Image
import time, random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--time', default=5, type=int, help='Time limit in minutes')
parser.add_argument('-l', '--level', default=3, type=int, help='Estimate Level')
args = parser.parse_args()

pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True

# detect the game board
print("Detecting the game board...")
x1, y1 = pyautogui.locateCenterOnScreen('top_left.png')[:2]
x2, y2 = pyautogui.locateCenterOnScreen('bottom_right.png')[:2]
print("Found board in the square (%d,%d) -> (%d,%d)" % (x1,y1,x2,y2))
print("Please do not move game window from now on.")
width, height = x2-x1, y2-y1


dsp = display.Display()
root = dsp.screen().root
def capture_screen(x, y, width, height):
    ''' A faster screen capture than the pyautogui.screenshot() '''
    raw = root.get_image(x, y, width, height, X.ZPixmap, 0xffffffff)
    image = Image.frombytes("RGB", (width, height), raw.data, "raw", "BGRX")
    return image

def capture_board_image():
    return capture_screen(x1, y1, width, height)



def read_game_state(image):
    black_stones, white_stones = set(), set()
    board_size = 15
    shift_x, shift_y = (width-1) / (board_size-1), (height-1) / (board_size-1)
    last_move = None
    playing = 0
    black_color = (44, 44, 44)
    grey_color = (220, 220, 220)
    white_color = (243, 243, 243)
    deep_black = (39, 39, 39)
    red_color = (253, 23, 30)
    for ir in xrange(15): # row
        for ic in xrange(15): # column
            stone = (ir+1, ic+1) # in the AI we count stone position starting from 1
            pos = (int(shift_x * ic), int(shift_y * ir))
            color = image.getpixel(pos)
            if color == black_color or color == grey_color: # black stone
                black_stones.add(stone)
            elif color == white_color or color == deep_black: # white stone
                white_stones.add(stone)
            elif color == red_color: # red square means just played
                # check the color of the new position
                newpos = (pos[0]+10, pos[1]) if ic < 14 else (pos[0]-10, pos[1])
                newcolor = image.getpixel(newpos)
                if newcolor == black_color: # black stone
                    black_stones.add(stone)
                    playing = 1 # white is playing next
                elif newcolor == white_color: # white stone
                    white_stones.add(stone)
                    playing = 0 # black is playing next
                else:
                    raise RuntimeError("Error when getting last played stone color.")
                last_move = stone
    board = (black_stones, white_stones)
    state = (board, last_move, playing, board_size)
    return state

def place_stone(move):
    board_size = 15
    ir, ic = move
    shift_x, shift_y = (width-1) / (board_size-1), (height-1) / (board_size-1)
    x = x1 + shift_x * (ic-1)
    y = y1 + shift_y * (ir-1)
    pyautogui.moveTo(x, y, duration=0.1)
    pyautogui.click()
    time.sleep(0.1)

def play_one_move(image, strategy):
    image = capture_board_image()
    state = read_game_state(image)
    print("Current Game Board:")
    print_state(state)
    print("Calculating next move...")
    next_move, q = strategy(state)
    winrate = ("%.1f%%" % ((q+1)/2*100)) if q != None else "??"
    print("Calculation finished. Playing (%d, %d) with win rate %s" % (next_move[0], next_move[1], winrate))
    place_stone(next_move)

def print_state(state):
    board, last_move, playing, board_size = state
    print(' '*4 + ' '.join([chr(97+i) for i in range(board_size)]))
    print(' '*3 + '='*(2*board_size))
    for x in range(1, board_size+1):
        row = ['%2s|'%x]
        for y in range(1, board_size+1):
            if (x,y) in board[0]:
                c = 'x'
            elif (x,y) in board[1]:
                c = 'o'
            else:
                c = '-'
            if (x,y) == last_move:
                c = '\033[92m' + c + '\033[0m'
            row.append(c)
        print(' '.join(row))

def game_paused(image):
    color1 = image.getpixel((343, 440))
    color2 = image.getpixel((540, 440))
    red_color = (236,43,36)
    if color1 == red_color and color2 == red_color:
        return True
    else:
        return False

def find_empty_place(image):
    board_size = 15
    shift_x, shift_y = (width-1) / (board_size-1), (height-1) / (board_size-1)
    black_color = (44, 44, 44)
    white_color = (243, 243, 243)
    red_color = (253, 23, 30)
    all_colors = set((black_color, white_color, red_color))
    for ir in xrange(15): # row
        for ic in xrange(15): # column
            stone = (ir+1, ic+1) # in the AI we count stone position starting from 1
            pos = (int(shift_x * ic), int(shift_y * ir))
            color = image.getpixel(pos)
            if color not in all_colors:
                return pos

def check_me_playing(maxtime=300):
    for _ in xrange(maxtime*2): # check every 0.5 s
        time.sleep(0.4)
        image = capture_board_image()
        state = read_game_state(image)
        board, last_move, playing, board_size = state
        if last_move != None or len(board[0]) == 0 or game_paused(image):
            break

def check_black_start():
    pyautogui.moveTo(x1+100, y1)
    time.sleep(0.1)
    image = capture_board_image()
    orig_color = image.getpixel((1, 1))
    pyautogui.moveTo(x1, y1)
    time.sleep(0.1)
    new_image = capture_board_image()
    new_color = new_image.getpixel((1, 1))
    if new_color != orig_color:
        return True
    else:
        return False

def click_start(image):
    global time_spent
    if image.getpixel((395, 466)) == (255,255,255):
        pyautogui.moveTo(x1+395, y1+466, duration=0.2)
        pyautogui.click()
        t_start = time.time()
        time_spent = 0
        player_AI.estimate_level = args.level
        for _ in xrange(20):
            time.sleep(0.5)
            image = capture_board_image()
            if game_paused(image) == False:
                if check_black_start():
                    time.sleep(0.5)
                    play_one_move(image, player_AI.strategy)
                    time_spent += time.time() - t_start
                    print("Total time spent: %.1f s" % time_spent)
                break



import construct_dnn
import player_AI
model = construct_dnn.construct_dnn()
model.load('tf_model')
player_AI.tf_predict_u.model = model
#player_AI.estimate_level = args.level
player_AI.initialize()

time_spent = 0
# loop to continue playing more games
while True:
    raw_input("Press Enter to start AI...")
    # loop to play multiple steps
    while True:
        try:
            time.sleep(0.5)
            image = capture_board_image()
            if game_paused(image):
                #print("Game Paused")
                time.sleep(1)
                click_start(image)
            else:
                # check if i'm playing, will wait here if not
                check_me_playing()
                # play a move
                t_start = time.time()
                image = capture_board_image()
                play_one_move(image, player_AI.strategy)
                time_spent += time.time() - t_start
                time_left = args.time * 60 - time_spent
                print("Time Left: %.1f s" % time_left)
                if time_left < 40 and player_AI.estimate_level > 2:
                    print("Switching to fast mode")
                    player_AI.estimate_level = 2
                if time_left < 20 and player_AI.estimate_level > 1:
                    print("Switching to ultrafast mode")
                    player_AI.estimate_level = 1
        except pyautogui.FailSafeException:
            print("Stopped by FailSafe")
            break
