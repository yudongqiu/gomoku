import tensorflow as tf
import tflearn
import random
import numpy as np

X = np.random.random([10,4,4,2])
Y = [[0] for x in X]

g1 = tf.Graph()

#with g1.as_default():
if True:
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

    m.load('m1')
#
#
# tf.reset_default_graph()
# input_layer = tflearn.input_data(shape=[None, 4, 4, 2])
# net = tflearn.conv_2d(input_layer, 128, 3, activation=None)
# net = tflearn.batch_normalization(net)
# net = tflearn.activation(net, activation='relu')
# # block 2
# tmp = tflearn.conv_2d(net, 128, 3, activation=None)
# tmp = tflearn.batch_normalization(tmp)
# net = tflearn.activation(net + tmp, activation='relu')
# final = tflearn.fully_connected(net, 1, activation='tanh')
# sgd = tflearn.optimizers.SGD(learning_rate=0.01, lr_decay=0.95, decay_step=200000)
# regression = tflearn.regression(final, optimizer=sgd, loss='mean_square',  metric='R2')
# m2 = tflearn.DNN(regression)


print(m.predict( X ))
#
# m2.load('test2')
# print(m2.pridict(X))
