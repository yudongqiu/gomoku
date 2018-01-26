#!/usr/bin/env python

from __future__ import division, print_function
from Xlib import display, X
from PIL import Image
import time, random
import pyautogui

pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True

class ScreenShot(object):
    """ This class can help quickly update a screenshot of certain region """
    @property
    def center(self):
        return int(0.5*(self.x2-self.x1)), int(0.5*(self.y2-self.y1))

    @property
    def border(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    def __init__(self, border=None):
        self.screen = display.Display().screen()
        self.root = self.screen.root
        self.update_border(border)

    def update_border(self, border):
        if border != None:
            self.x1, self.y1, self.x2, self.y2 = map(int, border)
            assert self.x2 > self.x1 and self.y2 > self.y1
        else:
            self.x1 = self.y1 = 0
            self.x2 = self.screen.width_in_pixels
            self.y2 = self.screen.height_in_pixels

    def capture(self):
        ''' A faster screen capture than the pyautogui.screenshot() '''
        raw = self.root.get_image(self.x1, self.y1, self.width, self.height, X.ZPixmap, 0xffffffff)
        image = Image.frombytes("RGB", (self.width, self.height), raw.data, "raw", "BGRX")
        return image

def read_game_state(scnshot):
    image = scnshot.capture()
    black_stones, white_stones = set(), set()
    board_size = 15
    shift_x, shift_y = (scnshot.width-1) / (board_size-1), (scnshot.height-1) / (board_size-1)
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
                newpos = (pos[0], pos[1]-15) if ir >0 else (pos[0], pos[1]+15)
                newcolor = image.getpixel(newpos)
                if newcolor == black_color: # black stone
                    black_stones.add(stone)
                    playing = 1 # white is playing next
                elif newcolor == white_color: # white stone
                    white_stones.add(stone)
                    playing = 0 # black is playing next
                else:
                    print("Error when getting last played stone color!")
                    print(newcolor,"at", newpos,"is not recognized!")
                    print("Trying one more time after 1s!")
                    time.sleep(1.0)
                    image = scnshot.capture()
                    newcolor = image.getpixel(newpos)
                    if newcolor == black_color:
                        black_stones.add(stone)
                        playing = 1
                    elif newcolor == white_color:
                        white_stones.add(stone)
                        playing = 0
                    else:
                        print("Error again!", newcolor,"at",newpos,"not recognized!")
                        image.save('debug.png')
                        print("Image saved to debug.png, exiting...")
                        raise RuntimeError
                last_move = stone
    board = (black_stones, white_stones)
    state = (board, last_move, playing, board_size)
    return state

def place_stone(scnshot, move):
    x1, y1, x2, y2 = scnshot.border
    board_size = 15
    ir, ic = move
    shift_x, shift_y = (scnshot.width-1) / (board_size-1), (scnshot.height-1) / (board_size-1)
    x = x1 + shift_x * (ic-1)
    y = y1 + shift_y * (ir-1)
    pyautogui.moveTo(x, y, duration=0.1)
    pyautogui.click()
    time.sleep(0.2)

def play_one_move(scnshot, strategy, verbose=True):
    t_start = time.time()
    state = read_game_state(scnshot)
    total_stones = get_total_stones(state)
    if verbose:
        print("Current Game Board:")
        print_state(state)
        print("Calculating next move...")
    next_move, q = strategy(state)
    if verbose:
        winrate = ("with win rate %.1f%%" % ((q+1)/2*100)) if q != None else "as the only choice."
        print("Calculation finished. Playing (%d, %d) %s" % (next_move[0], next_move[1], winrate))
    place_stone(scnshot, next_move)
    t_end = time.time()
    time_spent = 0
    # check if this play is successful, return the real time_spent
    new_state = read_game_state(scnshot)
    if get_total_stones(new_state) > total_stones:
        time_spent = t_end - t_start
        time.sleep(0.5) # give the website 0.5 s to process
    # else, we will return 0
    return time_spent

def get_total_stones(state):
    black_stones, white_stones = state[0]
    return len(black_stones) + len(white_stones)

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

def game_paused(scnshot):
    image = scnshot.capture()
    # find if the board is on the image
    found_board = False
    n_orange = 0
    board_color = (239, 175, 105)
    for x in range(5, 125, 10):
        for y in range(5, 125, 10):
            if image.getpixel((x,y)) == board_color:
                n_orange += 1
                if n_orange > 2:
                    found_board = True
        if found_board == True:
            break
    # if we don't find board in the image, return -1
    if found_board == False:
        return -1
    # check if the red bar is in the center
    cx, cy = scnshot.center
    n_red = 0
    red_color = (236,43,36)
    for x in range(cx-200, cx+200, 20):
        for y in range(cy-70, cy+70, 10):
            if image.getpixel((x,y)) == red_color:
                n_red += 1
                if n_red > 2:
                    return 1
    return 0

def check_me_playing(scnshot, maxtime=300):
    state = read_game_state(scnshot)
    board, last_move, playing, board_size = state
    if last_move != None: # if the opponent played
        return True
    else:
        return False

def click_start(scnshot):
    x1, y1, x2, y2 = scnshot.border
    cx, cy = scnshot.center
    white_color = (255,255,255)
    image = scnshot.capture()
    found_start = None
    for y in range(cy, cy+100, 5):
        if image.getpixel((cx,y)) == white_color:
            if image.getpixel((cx+40,y)) == white_color:
                if image.getpixel((cx+40,y+10)) == white_color:
                    found_start = (cx, y)
                    break
    game_started = False
    if found_start != None:
        x, y = found_start
        pyautogui.moveTo(x1+x, y1+y, duration=0.1)
        pyautogui.click()
        # wait for 10 s for opponent to click start
        for _ in xrange(20):
            time.sleep(0.5)
            if game_paused(scnshot) == False:
                game_started = True
                break
    return game_started

def swap_waiting(scnshot):
    "check if I'm choosing swap, return 0 if not, 1 if I'm first to play 3 stones, 2 if I'm able to choose side"
    x1, y1, x2, y2 = scnshot.border
    w, h = scnshot.width, scnshot.height
    cx, cy = scnshot.center
    image = scnshot.capture()
    white_color = (255,255,255)
    n_white = 0
    result = 0
    for x in range(cx-100, cx+100, 10):
        if image.getpixel((x,h-2)) == white_color:
            n_white += 1
            if n_white > 2:
                result = 1
                break
    n_white = 0
    shift = int((h-1) / 14)
    for x in range(cx-100, cx+100, 10):
        if image.getpixel((x,(h-shift))) == white_color:
            n_white += 1
            if n_white > 2:
                result = 2
                break
    return result

def choose_swap_start(scnshot, strategy):
    # wait for the swap label to appear
    swap_start = 0
    while swap_start == 0:
        swap_start = swap_waiting(scnshot)
        if game_paused(scnshot) != 0:
            return 0
        time.sleep(0.5)
    time_spent = 1.0
    if swap_start == 1: # if i'm first
        time_spent += place_first_three_stones(scnshot)
        # after placeing the first 3 stones, wait here until opponent made choice
        while True:
            time.sleep(0.5)
            swap_start = swap_waiting(scnshot)
            # if opponent chose to place two more stones
            if swap_start > 0:
                assert swap_start == 2
                time.sleep(0.5)
                time_spent += swap_choose_side(scnshot, strategy, expected=5)
                break
            # if opponent chose white or black, I should be getting my cursor back
            if read_game_state(scnshot)[1] != None: # if opponent place one move
                # we do one more check because the opponent might just placed one of the two stones
                if check_cursor_playing(scnshot):
                    break
    else: # if I'm second, choosing side then game starts
        time.sleep(0.5)
        time_spent += swap_choose_side(scnshot, strategy, expected=3)
    return time_spent


def check_cursor_playing(scnshot):
    x1, y1, x2, y2 = scnshot.border
    board, last_move, playing, board_size = read_game_state(scnshot)
    shift = int((scnshot.width - 1) / 14)
    # we should be able to find an empty spot on the first row
    for c in xrange(10):
        if (0,c) not in board[0] and (0,c) not in board[1]:
            empty_place = c
            break
    empty_pos = (c, 0)
    # move mouse to third row
    pyautogui.moveTo(x1, y1+shift*2, duration=0.1)
    time.sleep(0.2)
    orig_color = scnshot.capture().getpixel(empty_pos)
    pyautogui.moveTo(x1+c*shift, y1, duration=0.2)
    new_color = scnshot.capture().getpixel(empty_pos)
    return not (new_color == orig_color)



def place_first_three_stones(scnshot):
    t_start = time.time()
    o1 = (4,4), (6,7), (8,8)
    o2 = (3,6), (7,7), (6,8)
    o3 = (7,8), (9,10),(10,12)
    o4 = (11,6),(8,10),(9,8)
    #o5 = (9,3), (5,12),(4,6)
    openings = [o1, o2, o3, o4]#, o5]
    openmoves = random.choice(openings)
    print("Playing first three moves!", openmoves)
    for move in openmoves:
        place_stone(scnshot, move)
    return time.time() - t_start

def swap_choose_side(scnshot, strategy, expected=3):
    assert expected in (3,5)
    # number of black and white stones expected
    eb, ew = (2,1) if expected == 3 else (3,2)
    t_start = time.time()
    x1, y1, x2, y2 = scnshot.border
    w, h = scnshot.width, scnshot.height
    state = read_game_state(scnshot)
    print_state(state)
    (black_stones, white_stones), last_move, playing, board_size = state
    black_button = (int(w/2-20) + x1, y2)
    white_button = (int(w/2+20) + x1, y2)
    if len(black_stones) == 0:
        # if I don't see any black stones
        # this shouldn't happen unless the stones are hidden behide label
        print("Choosing white side because I don't see black stones!")
        pyautogui.click(white_button)
    else:
        stone_not_seen = False
        if len(black_stones) != eb or len(white_stones) != ew:
            print("Warning! I'm not seeing all stones, double checking")
            time.sleep(0.7)
            state = read_game_state(scnshot)
            (black_stones, white_stones), last_move, playing, board_size = state
            nb, nw = len(black_stones), len(white_stones)
            if nb != eb or nw != ew:
                print("Still can't see all stones, very likely they're blocked by white bar!")
                for i in range(eb-nb):
                    black_stones.add((14,7+i))
                    print("Assuming there's a black stone at (14,%d)" % (i+7))
                for i in range(ew-nw):
                    white_stones.add((14,9+i))
                    print("Assuming there's a white stone at (14,%d)" % (i+9))
                state = (black_stones, white_stones), last_move, playing, board_size
                print("Final assumed state:")
                print_state(state)
                stone_not_seen = True
        # replace last move with any black stones
        last_move = next(iter(black_stones))
        # set player to white
        playing = 1
        state = (black_stones, white_stones), last_move, playing, board_size
        next_move, q = strategy(state)
        wr = (q+1)/2*100
        if q > 0:
            print("Choosing white side! Win Rate: %.1f%%" % wr)
            pyautogui.click(white_button)
            time.sleep(0.5)
            if stone_not_seen == True:
                print("Seeing the actual state")
                state = read_game_state(scnshot)
                print_state(state)
                next_move, q = strategy(state)
                print("playing (%d,%d) with win rate %.1f%%" % (next_move[0], next_move[1], wr))
            place_stone(scnshot, next_move)
        else:
            print("Choosing black side! Win Rate: %.1f%%" % (100-wr))
            pyautogui.click(black_button)
    time_spent = time.time() - t_start
    time.sleep(0.5)
    return time_spent



def detect_board_edge():
    try:
        x1, y1 = pyautogui.locateCenterOnScreen('top_left.png')[:2]
        x2, y2 = pyautogui.locateCenterOnScreen('bottom_right.png')[:2]
    except:
        raise RuntimeError("Board not found on the screen!")
    return x1, y1, x2, y2


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Player Gomoku on playok.com', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--time', default=5, type=int, help='Time limit in minutes')
    parser.add_argument('-l', '--level', default=3, type=int, help='Estimate Level')
    parser.add_argument('-d', '--detect', default=False, action='store_true', help='Detect game board at beginning')
    args = parser.parse_args()

    if args.detect:
        # detect the game board
        print("Detecting the game board...")
        x1, y1, x2, y2 = detect_board_edge()
    else:
        x1, y1, x2, y2 = (2186,237,3063,1114)
    print("Set board in the square (%d,%d) -> (%d,%d)" % (x1,y1,x2,y2))
    print("Please do not move game window from now on.")

    scnshot = ScreenShot(border=(x1,y1,x2,y2))
    # load the AI player
    import construct_dnn
    import AI_Swap
    model = construct_dnn.construct_dnn()
    model.load('tf_model')
    AI_Swap.tf_predict_u.model = model
    AI_Swap.initialize()

    time_spent = 0
    total_time = args.time * 60
    # loop to play multiple steps
    while True:
        try:
            time.sleep(0.5)
            status = game_paused(scnshot)
            if status == -1: # game board not found
                time.sleep(1)
            elif status == 1:
                time.sleep(1)
                # try to click the start button and wait for game start
                if click_start(scnshot) == True:
                    # if game started, we check if we are the black first
                    AI_Swap.estimate_level = args.level
                    print("Game started with AI level = %d" % args.level)
                    time_spent = choose_swap_start(scnshot, AI_Swap.strategy)
                    AI_Swap.reset()
                    print("Time Left: %02d:%02d " % divmod(total_time - time_spent, 60))
            else:
                # check if i'm playing, will wait here if not
                if check_me_playing(scnshot) == True:
                    time_spent += play_one_move(scnshot, AI_Swap.strategy)
                    # check how much time left
                    time_left = total_time - time_spent
                    print("Time Left: %02d:%02d " % divmod(time_left, 60))
                    tdown2 = min(total_time*0.6, 60)
                    if time_left < tdown2 and AI_Swap.estimate_level > 2:
                        print("Switching to fast mode, AI level = 2")
                        AI_Swap.estimate_level = 2
                    tdown1 = min(total_time*0.3, 30)
                    if time_left < tdown1 and AI_Swap.estimate_level > 1:
                        print("Switching to ultrafast mode, AI level = 1")
                        AI_Swap.estimate_level = 1
        except (KeyboardInterrupt, pyautogui.FailSafeException):
            new_total_time = raw_input("Stopped by user, enter new time limit in minutes, or enter to continue...")
            try:
                total_time = float(new_total_time)*60
                print("New total time has been set to %.1f seconds" % total_time)
            except:
                pass

if __name__ == '__main__':
    main()
