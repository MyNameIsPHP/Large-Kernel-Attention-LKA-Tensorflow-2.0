from keras.layers import Layer
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, regularizers, Input, Model

class LargeKernelAttnLayer(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(LargeKernelAttnLayer, self).__init__(**kwargs)
        self.channels = channels
        self.dwconv = layers.DepthwiseConv2D(kernel_size=5, padding='same', depth_multiplier=1)
        self.dwdconv = layers.DepthwiseConv2D(kernel_size=7, padding='same', depth_multiplier=1, dilation_rate=3)
        self.pwconv = layers.Conv2D(filters=self.channels, kernel_size=1)

    def call(self, inputs):
        weight = self.pwconv(self.dwdconv(self.dwconv(inputs)))
        return inputs * weight

    def get_config(self):
        config = super(LargeKernelAttnLayer, self).get_config()
        config.update({"channels": self.channels})
        return config
