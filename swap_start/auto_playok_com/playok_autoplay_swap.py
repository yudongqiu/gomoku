#!/usr/bin/env python

from __future__ import division, print_function
from Xlib import display, X
from PIL import Image
import time, random
import pyautogui

pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True

class Color:
    def __init__(self, r, g, b, t=10):
        self.rgb = (r, g, b)
        self.t = t

    def __eq__(self, otherrgb):
        if isinstance(otherrgb, Color):
            otherrgb = otherrgb.rgb
        for c1, c2 in zip(self.rgb, otherrgb):
            if abs(c1 - c2) > self.t:
                return False
        return True

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
    for ir in range(15): # row
        for ic in range(15): # column
            stone = (ir+1, ic+1) # in the AI we count stone position starting from 1
            # center pos
            pos = (int(shift_x * ic), int(shift_y * ir))
            stone_type = get_stone_type(image, pos)
            if stone_type is not None:
                if stone_type == 'b':
                    black_stones.add(stone)
                elif stone_type == 'bl':
                    black_stones.add(stone)
                    # white is playing next
                    playing = 1
                    last_move = stone
                elif stone_type == 'w':
                    white_stones.add(stone)
                elif stone_type == 'wl':
                    white_stones.add(stone)
                    # black is playing next
                    playing = 0
                    last_move = stone
    board = (black_stones, white_stones)
    state = (board, last_move, playing, board_size)
    return state

def get_stone_type(image, pos):
    """ Get the stone type at pos,
    return 'b' for black, 'bl' for black last played,
           'w' for white, 'wl' for white last played, None if not found """
    black_color = Color(44, 44, 44)
    white_color = Color(243, 243, 243)
    w, h = image.size
    x, y = pos
    all_pos = [(x+dx, y+dy) for dx, dy in [(-15, 3), (15, -2), (4, 15), (-3, -15)] if x+dx >= 0 and x+dx <= w and y+dy >= 0 and y+dy <= h]
    if all(image.getpixel(p) == black_color for p in all_pos):
        if image.getpixel(pos) != black_color:
            ret = 'bl'
        else:
            ret = 'b'
    elif all(image.getpixel(p) == white_color for p in all_pos):
        if image.getpixel(pos) != white_color:
            ret = 'wl'
        else:
            ret = 'w'
    else:
        ret = None
    return ret

def place_stone(scnshot, move):
    x1, y1, x2, y2 = scnshot.border
    board_size = 15
    ir, ic = move
    shift_x, shift_y = (scnshot.width-1) / (board_size-1), (scnshot.height-1) / (board_size-1)
    x = x1 + shift_x * (ic-1)
    y = y1 + shift_y * (ir-1)
    pyautogui.moveTo(x, y, duration=0.1)
    pyautogui.click()
    time.sleep(0.3)

def play_one_move(scnshot, strategy, verbose=True):
    t_start = time.time()
    for _ in range(10):
        state = read_game_state(scnshot)
        if get_total_stones(state) == 0:
            break
        else:
            last_move = state[1]
            if last_move is None:
                print("Did not find last move, rechecking...")
                time.sleep(0.5)
            else:
                break
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
    board_color = Color(235, 178, 108)
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
    red_color = Color(226, 61, 37)
    for x in range(cx-200, cx+200, 20):
        for y in range(cy-70, cy+70, 10):
            if image.getpixel((x,y)) == red_color:
                n_red += 1
                if n_red > 4:
                    return 1
    return 0

def check_me_playing(scnshot2):
    w, h = scnshot2.width, scnshot2.height
    image = scnshot2.capture()
    start = max(w-20, 0)
    playing_color = Color(31, 41, 47)
    my_turn = False
    for posx in range(start, w-3, 2):
        if image.getpixel((posx,h-1)) == playing_color and image.getpixel((posx+2,h-2)) == playing_color:
            my_turn = True
            break
    if my_turn == True:
        c2 = image.getpixel((0,0))
        if c2 == Color(255,255,255):
            return 1 # white player
        else:
            return 0
    else:
        return None

def click_start(scnshot):
    x1, y1, x2, y2 = scnshot.border
    cx, cy = scnshot.center
    white_color = Color(255,255,255)
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
        for _ in range(20):
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
    white_color = Color(255,255,255)
    n_white = 0
    result = 0
    shift = int((h-1) / 14)
    for x in range(cx-shift*2, cx+shift*2, 10):
        if image.getpixel((x,h-2)) == white_color:
            n_white += 1
            if n_white > 2:
                result = 1
                break
    n_white = 0
    for x in range(cx-shift*2, cx+shift*2, 10):
        if image.getpixel((x,(h-shift))) == white_color:
            n_white += 1
            if n_white > 2:
                result = 2
                break
    return result

def choose_swap_start(scnshot, scnshot2, strategy):
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
                if swap_start != 2: continue
                time.sleep(0.5)
                time_spent += swap_choose_side(scnshot, strategy, expected=5)
                break
            # if opponent chose white or black, I will continue to play regular
            elif check_me_playing(scnshot2) != None:
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
    for c in range(10):
        if (0,c) not in board[0] and (0,c) not in board[1]:
            empty_place = c
            break
    empty_pos = (c*shift, 0)
    # move mouse to third row
    pyautogui.moveTo(x1, y1+shift*2, duration=0.1)
    time.sleep(0.2)
    orig_color = scnshot.capture().getpixel(empty_pos)
    pyautogui.moveTo(x1+c*shift, y1, duration=0.1)
    time.sleep(0.2)
    new_color = scnshot.capture().getpixel(empty_pos)
    return not (new_color == orig_color)

begin_lib = [[ ( 8, 8),  ( 7, 9), (11,11)],
             [ (11, 5),  ( 8, 7), (10, 9)],
             [ ( 8, 6),  ( 5, 8), ( 8,10)],
             [ ( 8, 6),  ( 6, 8), ( 8,10)],
             [ (10, 7),  ( 4,11), (10,13)]]

def place_first_three_stones(scnshot):
    t_start = time.time()
    openings = begin_lib
    openmoves = random.choice(openings)
    print("Playing first three moves!", openmoves)
    for move in openmoves:
        place_stone(scnshot, move)
        time.sleep(0.5)
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
        if stone_not_seen == True:
            q = rough_estimate_q(state)
        else:
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

def rough_estimate_q(state):
    orig_level = rough_estimate_q.AI.estimate_level
    rough_estimate_q.AI.estimate_level = 2
    _, q = rough_estimate_q.AI.strategy(state)
    rough_estimate_q.AI.estimate_level = orig_level
    return q

def detect_board_edge():
    try:
        x1, y1 = pyautogui.locateCenterOnScreen('top_left.png', confidence=0.95)[:2]
        x2, y2 = pyautogui.locateCenterOnScreen('bottom_right.png', confidence=0.95)[:2]
    except:
        raise RuntimeError("Board not found on the screen!")
    return x1, y1, x2, y2

def detect_side_location():
    try:
        x, y = pyautogui.locateCenterOnScreen('tl2.png')[:2]
    except:
        raise RuntimeError("Side window not found on the screen!")
    return x+17, y-13, x+111, y+117

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
        b1 = detect_board_edge()
        b2 = detect_side_location()
    else:
        #x1, y1, x2, y2 = (2186,237,3063,1114)
        #b2 = (3245, 300, 3315, 400)
        b1 = (2287, 269, 3007, 989)
        b2 = (3151, 333, 3220, 429)
    print("Set board in the square (%d,%d) -> (%d,%d)" % b1)
    print("Please do not move game window from now on.")

    scnshot = ScreenShot(border=b1)
    # 2nd scnshot for checking me playing
    scnshot2 = ScreenShot(border=b2)
    # load the AI player
    from tf_model import load_existing_model
    import AI_Swap
    model = load_existing_model('tf_model.h5')
    AI_Swap.tf_predict_u.model = model
    AI_Swap.initialize()

    rough_estimate_q.AI = AI_Swap

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
                    time_spent = choose_swap_start(scnshot, scnshot2, AI_Swap.strategy)
                    AI_Swap.reset()
                    print("Time Left: %02d:%02d " % divmod(total_time - time_spent, 60))
            else:
                # check if i'm playing, will wait here if not
                playing = check_me_playing(scnshot2)
                if playing != None:
                    board, last_move, state_playing, board_size = read_game_state(scnshot)
                    if playing != state_playing:
                        print("Warning: The current player is not consistent! Rechecking state ...")
                        continue
                    if last_move == None:
                        print("Warning: Did not find last move! Rechecking state ...")
                        continue
                    t = play_one_move(scnshot, AI_Swap.strategy)
                    if t is None: continue
                    time_spent += t
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
            new_total_time = input("Stopped by user, enter new time limit in minutes, or enter to continue...")
            try:
                total_time = float(new_total_time)*60
                print("New total time has been set to %.1f seconds" % total_time)
            except:
                pass

if __name__ == '__main__':
    main()
