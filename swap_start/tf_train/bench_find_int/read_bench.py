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
            s = s.replace('x','1')
            s = s.replace('o','-1')
            state.append(s.split())
    print(state)
    return np.array(state, dtype=np.int8)

state = read_state('state_debug.txt')
player_A.print_state(state)

last_move = (4,11)

player_A.print_state(state)

import construct_dnn
model = construct_dnn.construct_dnn()

player_A.initialize()
player_A.strategy( (({(1,1)},{}),(1,1),1,15) )
player_A.tf_predict_u.model = model 
model.load('../../auto_playok_com/tf_model')

player_A.best_action_q(state, 12342, 180, last_move, -1, 1, 1, -1)

import IPython
IPython.embed()

#player_A.find_interesting_moves(state, 120, np.zeros((15,15)), 1, 50, True)
