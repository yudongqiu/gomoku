# coding: utf-8
import numpy as np
import player_A
state = np.zeros((15,15), dtype=np.int8)
state[8,2] = state[3,5] = state[5,5] = 1
state[4,10] = state[4,11] = -1
player_A.print_state(state)
player_A.find_interesting_moves(state, 220, np.zeros((15,15)), -1, 20, True)
