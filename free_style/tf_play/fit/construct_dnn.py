import tflearn

def construct_dnn():
    input_layer = tflearn.input_data(shape=[None, 5, 15, 15])
    net = tflearn.fully_connected(input_layer, 384, activation='linear')
    net = tflearn.fully_connected(net, 192, activation='relu')
    #net = tflearn.fully_connected(net, 192, activation='tanh')
    net = tflearn.fully_connected(net, 192, activation='relu')
    #net = tflearn.fully_connected(net, 192, activation='tanh')
    #net = tflearn.fully_connected(net, 192, activation='relu')
    #net = tflearn.fully_connected(net, 192, activation='tanh')
    #net = tflearn.fully_connected(net, 192, activation='relu')
    final = tflearn.fully_connected(net, 1, activation='sigmoid')
    sgd = tflearn.optimizers.SGD(learning_rate=0.1, lr_decay=0.99, decay_step=10000) 
    regression = tflearn.regression(final, optimizer=sgd, loss='mean_square',  metric='R2')
    model = tflearn.DNN(regression)
    return model
