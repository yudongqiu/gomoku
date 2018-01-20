# coding: utf-8
import numpy as np
import h5py
h5f = h5py.File('data.h5', 'r')
tx = h5f['bx'][300:320]
nx = len(tx)
testx = np.empty(nx*15*15*5, dtype=np.int8).reshape(nx,15,15,5)
testx[:, :, :, 0] = (tx == 1)
testx[:, :, :, 1] = (tx == -1)
testx[:, :, :, 2] = (tx == 0)
testx[:, :, :, 3] = 1
testx[:, :, :, 4] = 1
testy = h5f['train_Y'][300:320]
def draw_state(state):
    board_size = 15
    print(' '*4 + ' '.join([chr(97+i) for i in xrange(board_size)]))
    print (' '*3 + '='*(2*board_size))
    me = state[0,0,4]
    for x in xrange(1, board_size+1):
        row = ['%2s|'%x]
        for y in xrange(1, board_size+1):
            if state[x-1,y-1,0] == 1:
                c = 'x' if me == 1 else 'o'
            elif state[x-1,y-1,1] == 1:
                c = 'o' if me == 1 else 'x'
            else:
                c = '-'
            row.append(c)
        print (' '.join(row))
        
import construct_dnn
model = construct_dnn.construct_dnn()
model.load('tf_model')
py = model.predict(testx)
for i in range(len(testx)):
    draw_state(testx[i])
    print("ref        y : %.5f"%testy[i][0])
    print("predicted  y : %.5f"%py[i][0])
    
    
