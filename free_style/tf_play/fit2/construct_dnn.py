import tflearn

def construct_dnn():
    input_layer = tflearn.input_data(shape=[None, 15, 15, 5])
    net = tflearn.conv_2d(input_layer, 192, 3, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.fully_connected(net, 192, activation='relu')
    final = tflearn.fully_connected(net, 1, activation='sigmoid')
    sgd = tflearn.optimizers.SGD(learning_rate=0.1, lr_decay=0.99, decay_step=10000)
    regression = tflearn.regression(final, optimizer=sgd, loss='mean_square',  metric='R2')
    model = tflearn.DNN(regression)
    return model
