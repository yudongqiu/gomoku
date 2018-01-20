# coding: utf-8
import tensorflow as tf
import tflearn
import numpy as np
import pickle

def build_train_data():

    bdata = pickle.load(open('black.learndata','rb'))
    bx, by, bn = zip(*bdata.values())
    bx = np.array(bx)
    by = np.array(by)
    nb = len(bx)
    print("Successfully loaded %d black trained data" % nb)
    wdata = pickle.load(open('white.learndata','rb'))
    wx, wy, wn = zip(*wdata.values())
    wx = np.array(wx)
    wy = np.array(wy)
    nw = len(wx)
    print("Successfully loaded %d white trained data" % nw)
    n_data = nb+nw

    train_X = np.empty(n_data*5*15*15, dtype=np.int32).reshape(n_data, 5, 15, 15)
    # fill the black.data part
    train_X[:nb, 0, :, :] = (bx == 1) # first plane indicates my stones
    train_X[:nb, 1, :, :] = (bx == -1) # first plane indicates opponent_stones
    train_X[:nb, 2, :, :] = (bx == 0) # third plane indicates empty spots
    train_X[:nb, 3, :, :] = 1 # fourth plane is all 1
    train_X[:nb, 4, :, :] = 1 # 5th plane indicates if i'm black
    # fill the white.data part
    train_X[nb:, 0, :, :] = (wx == 1) # first plane indicates my stones
    train_X[nb:, 1, :, :] = (wx == -1) # first plane indicates opponent_stones
    train_X[nb:, 2, :, :] = (wx == 0) # third plane indicates empty spots
    train_X[nb:, 3, :, :] = 1 # fourth plane is all 1
    train_X[nb:, 4, :, :] = 0 # 5th plane indicates if i'm black

    train_Y = np.empty(n_data, dtype=np.float32)
    train_Y[:nb] = by
    train_Y[nb:] = wy
    train_Y = train_Y.reshape(-1,1)

    print("Trainning data built!")
    return train_X, train_Y

#train_X, train_Y = build_train_data()
import h5py
h5f = h5py.File('data.h5', 'r')
train_X = np.array(h5f['train_X'])[:]
train_Y = np.array(h5f['train_Y'])[:]

import construct_dnn
model = construct_dnn.construct_dnn()

model.fit(train_X, train_Y, n_epoch=150, validation_set=0.1, show_metric=True)
model.save("tf_model")
print("Model saved to tf_model!")

h5f.close()
