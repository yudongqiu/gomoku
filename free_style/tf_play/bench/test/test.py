import tensorflow as tf
import tflearn
import random
import numpy as np

#X = [[random.random(),random.random()] for x in range(1000)]
X = np.random.random([10,4,4,2])
Y = [[0] for x in X]

g1 = tf.Graph()
g2 = tf.Graph()

with g1.as_default():
  input_layer = tflearn.input_data(shape=[None, 4, 4, 2])
  net = tflearn.conv_2d(input_layer, 256, 3, activation=None)
  net = tflearn.batch_normalization(net)
  net = tflearn.activation(net, activation='relu')
  # block 2
  tmp = tflearn.conv_2d(net, 256, 3, activation=None)
  tmp = tflearn.batch_normalization(tmp)
  tmp = tflearn.activation(tmp, activation='relu')
  tmp = tflearn.conv_2d(tmp, 256, 3, activation=None)
  tmp = tflearn.batch_normalization(tmp)
  net = tflearn.activation(net + tmp, activation='relu')
  final = tflearn.fully_connected(net, 1, activation='tanh')
  sgd = tflearn.optimizers.SGD(learning_rate=0.01, lr_decay=0.95, decay_step=200000)
  regression = tflearn.regression(final, optimizer=sgd, loss='mean_square',  metric='R2')
  m = tflearn.DNN(regression)

with g2.as_default():
  input_layer = tflearn.input_data(shape=[None, 4, 4, 2])
  net = tflearn.conv_2d(input_layer, 128, 3, activation=None)
  net = tflearn.batch_normalization(net)
  net = tflearn.activation(net, activation='relu')
  # block 2
  tmp = tflearn.conv_2d(net, 128, 3, activation=None)
  tmp = tflearn.batch_normalization(tmp)
  net = tflearn.activation(net + tmp, activation='relu')
  final = tflearn.fully_connected(net, 1, activation='tanh')
  sgd = tflearn.optimizers.SGD(learning_rate=0.01, lr_decay=0.95, decay_step=200000)
  regression = tflearn.regression(final, optimizer=sgd, loss='mean_square',  metric='R2')
  m2 = tflearn.DNN(regression)

m.fit(X, Y, n_epoch = 10)
print(m.predict( [X[0]] ))
with g1.as_default():
  m.save('m1')

m2.fit(X, Y, n_epoch = 10)
print(m.predict( [X[0]] ))
with g2.as_default():
  m2.save('m2')
