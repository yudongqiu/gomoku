import os
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("datafile", default='data.h5', help='Path to data.h5')
parser.add_argument('-m', "--model", default='tf_model', help='Path to tf_model')
parser.add_argument('-n', '--n_epoch', default=100, type=int, help='n_epoch of training')
args = parser.parse_args()


h5f = h5py.File(args.datafile)
bx, by = np.array(h5f['bx']), np.array(h5f['by'])
wx, wy = np.array(h5f['wx']), np.array(h5f['wy'])

nb, nw = len(bx), len(wx)
n_data = nb + nw
train_X = np.empty(n_data*15*15*3, dtype=np.int8).reshape(n_data,15,15,3)
train_Y = np.empty(n_data, dtype=np.float32)

train_X[:nb, :, :, 0] = (bx == 1)
train_X[:nb, :, :, 1] = (bx == -1)
train_X[:nb, :, :, 2] = 1

train_X[nb:, :, :, 0] = (wx == 1)
train_X[nb:, :, :, 1] = (wx == -1)
train_X[nb:, :, :, 2] = 0

train_Y[:nb] = by
train_Y[nb:] = wy
train_Y = train_Y.reshape(-1,1)

import construct_dnn
model = construct_dnn.construct_dnn()
model.load(args.model)

os.mkdir('trained_model')
os.chdir('trained_model')

model.fit(train_X, train_Y, n_epoch=args.n_epoch, validation_set=0.1, show_metric=True)
model.save('tf_model')
print("New model saved!")
