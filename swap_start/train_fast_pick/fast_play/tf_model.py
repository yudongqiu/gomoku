#!/usr/bin/env python
# coding: utf-8

import sys
import random
import tensorflow as tf
from tensorflow.keras import layers
tf.config.gpu.set_per_process_memory_fraction(0.4)
tf.keras.backend.clear_session()


class ResNetBlock(layers.Layer):
    def __init__(self, filters=256, kernel_size=3):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        bn_axis = 3
        # build layers
        self.conv1 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.batch1 = layers.BatchNormalization(axis=bn_axis)
        self.activ1 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.batch2 = layers.BatchNormalization(axis=bn_axis)
        self.activ2 = layers.Activation('relu')
        self.add = layers.Add()

    def call(self, inputs, training=None):
        x = inputs
        x = self.conv1(x)
        x = self.batch1(x, training=training)
        x = self.activ1(x)
        x = self.conv2(x)
        x = self.batch2(x, training=training)
        x = self.activ2(x)
        x = self.add([x, inputs])
        return x

    def get_config(self):
        return {'filters': self.filters, 'kernel_size': self.kernel_size}

class DNNModel(tf.keras.Model):

    def __init__(self, n_stages=4, filters=256, kernel_size=3):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        bn_axis = 3
        # build model network
        self.layer_input = layers.InputLayer(input_shape=(15,15,3), name='input', dtype='float16')
        self.layer_conv0 = layers.Conv2D(self.filters, self.kernel_size, padding='same')
        self.layer_batch0 = layers.BatchNormalization(axis=bn_axis)
        self.layer_activ0 = layers.Activation('relu')
        # a list of resnet blocks
        self.layer_resBlocks = [ResNetBlock(filters=self.filters, kernel_size=self.kernel_size) for _ in range(n_stages)]
        # final evaluation head
        self.layer_final_conv = layers.Conv2D(1, (1, 1))
        self.layer_final_batch = layers.BatchNormalization(axis=bn_axis)
        self.layer_final_activ = layers.Activation('relu')
        self.layer_flatten = layers.Flatten()
        self.layer_dense = layers.Dense(256, activation='relu')
        self.layer_res = layers.Dense(1, activation='tanh', name='result')

    def call(self, inputs, training=False):
        x = inputs
        if training:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            x = tf.image.rot90(x, k=random.randint(0,3))
        x = self.layer_input(x)
        x = self.layer_conv0(x)
        x = self.layer_batch0(x, training=training)
        x = self.layer_activ0(x)
        for res_block in self.layer_resBlocks:
            x = res_block(x, training=training)
        x = self.layer_final_conv(x)
        x = self.layer_final_batch(x, training=training)
        x = self.layer_final_activ(x)
        x = self.layer_flatten(x)
        x = self.layer_dense(x)
        res = self.layer_res(x)
        return res

    # def save(self, path):
    #     """ Customized save method
    #     This is needed because the default save method only support Sequential/Functional Model
    #     """
    #     # Save JSON config to disk
    #     json_config = model.to_json()
    #     with open('model_config.json', 'w') as json_file:
    #         json_file.write(json_config)
    #     # Save weights to disk
    #         model.save_weights('path_to_my_weights.h5')

    # def load(self, path):
    #     # Reload the model from the 2 files we saved
    #     with open('model_config.json') as json_file:
    #         json_config = json_file.read()
    #     new_model = keras.models.model_from_json(json_config)
    #     self.load_weights('path_to_my_weights.h5')



def get_new_model():
    model = DNNModel(n_stages=20, filters=256, kernel_size=5)
    optimizer = tf.optimizers.Adagrad(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def save_model(model, path):
    model.save_weights(path)

def load_existing_model(path):
    model = get_new_model()
    model.load_weights(path)
    return model

#
def test_model():
    import numpy as np
    x_train = np.random.randint(0, 1, size=(1000,15,15,3)).astype(np.float32)
    y_train = np.random.random(1000)*2-1
    model = get_new_model()
    model.fit(x_train, y_train, epochs=10, validation_split=0.2)
    model.summary()
    model.evaluate(x_train, y_train)

if __name__ == '__main__':
    test_model()
