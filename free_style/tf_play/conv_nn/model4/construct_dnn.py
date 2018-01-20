import tflearn

def construct_dnn():
    tflearn.init_graph(num_cores=4, gpu_memory_fraction=0.5)
    img_aug = tflearn.ImageAugmentation()
    img_aug.add_random_90degrees_rotation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_flip_updown()
    input_layer = tflearn.input_data(shape=[None, 15, 15, 5], data_augmentation=img_aug)
    net = tflearn.conv_2d(input_layer, 256, 5, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.fully_connected(net, 192, activation='relu')
    final = tflearn.fully_connected(net, 1, activation='sigmoid')
    sgd = tflearn.optimizers.SGD(learning_rate=0.1, lr_decay=0.95, decay_step=200000)
    regression = tflearn.regression(final, optimizer=sgd, loss='mean_square',  metric='R2')
    model = tflearn.DNN(regression)#, tensorboard_verbose=3)
    return model
