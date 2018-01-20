import tflearn

def construct_dnn():
    input_layer = tflearn.input_data(shape=[None, 5, 15, 15])
    net = tflearn.fully_connected(input_layer, 384, activation='linear')
    net = tflearn.fully_connected(net, 256, activation='relu')
    net = tflearn.fully_connected(net, 64, activation='relu')
    net = tflearn.fully_connected(net, 32, activation='relu')
    final = tflearn.fully_connected(net, 1, activation='tanh')
    regression = tflearn.regression(final, optimizer='SGD', learning_rate=0.01, loss='mean_square',  metric='R2')
    model = tflearn.DNN(regression)
    return model
