# coding: utf-8
import tensorflow as tf
import tflearn
import numpy as np
import pickle

#train_X, train_Y = build_train_data()
import h5py
h5f = h5py.File('data.h5', 'r')
bx = np.array(h5f['bx'], dtype=np.int8)
wx = np.array(h5f['wx'], dtype=np.int8)
nb = len(bx)
nw = len(wx)
n_data = nb + nw
train_Y = h5f['train_Y'][:]
train_Y = np.array(train_Y) * 2 - 1
train_X = np.empty(n_data*15*15*3, dtype=np.int8).reshape(n_data,15,15,3)
# fill the black.data part
train_X[:nb, :, :, 0] = (bx == 1) # first plane indicates my stones
train_X[:nb, :, :, 1] = (bx == -1) # first plane indicates opponent_stones
train_X[:nb, :, :, 2] = 1 # third plane indicates if i'm black
# fill the white.data part
train_X[nb:, :, :, 0] = (wx == 1) # first plane indicates my stones
train_X[nb:, :, :, 1] = (wx == -1) # first plane indicates opponent_stones
train_X[nb:, :, :, 2] = 0 # third plane indicates if i'm black


import construct_dnn
model = construct_dnn.construct_dnn()

model.fit(train_X, train_Y, n_epoch=100, validation_set=0.1, show_metric=True)
model.save("tf_model")
print("Model saved to tf_model!")

h5f.close()
