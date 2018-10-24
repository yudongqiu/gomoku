# coding: utf-8
import numpy as np
import AI_Swap

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
AI_Swap.print_state(state)

last_move = (4,11)

AI_Swap.print_state(state)

import construct_dnn
model = construct_dnn.construct_dnn()

model.load('../../auto_playok_com/tf_model')

AI_Swap.initialize()

AI_Swap.tf_predict_u.model = model 

AI_Swap.strategy( (({(1,1)},{}),(1,1),1,15) )

AI_Swap.best_action_q(state, 12342, 180, last_move, -1, 1, 1, -1)

import IPython
IPython.embed()

#AI_Swap.find_interesting_moves(state, 120, np.zeros((15,15)), 1, 50, True)
