# coding: utf-8
import numpy as np
import player_A

def read_state(f):
    reading = False
    state = []#np.zeros((15,15), dtype=np.int8)
    for line in open(f):
        if "====" in line:
            reading = True
        elif reading == True:
            if '|' not in line: break
            s = line[4:].strip()
            s = s.replace('-','0')
            s = s.replace('x','-1')
            s = s.replace('o','1')
            state.append(s.split())
    print(state)
    return np.array(state, dtype=np.int8)

state = read_state('state.txt')
player_A.print_state(state)



player_A.print_state(state)

import IPython
IPython.embed()

player_A.find_interesting_moves(state, 120, np.zeros((15,15)), 1, 50, True)
