# coding: utf-8
import h5py
h5f = h5py.File('../fit/data.h5', 'r')
testx = h5f['train_X'][300:320].transpose(0,2,3,1)
testy = h5f['train_Y'][300:320]
def draw_state(state):
    board_size = 15
    print(' '*4 + ' '.join([chr(97+i) for i in xrange(board_size)]))
    print (' '*3 + '='*(2*board_size))
    me = state[4,0,0]
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
    
    
