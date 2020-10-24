import tensorflow as tf
from tensorflow.keras.layers import Flatten


class DistFlatten(tf.keras.layers.Layer):

    def __init__(self):
        super(DistFlatten, self).__init__()
        self.flatten_layer = Flatten()


    def call(self, x):
        if isinstance(x, tuple):
            assert len(x) == 2
            x_mean, x_var = x[0], x[1]
            x_mean = self.flatten_layer(x_mean)
            x_var = self.flatten_layer(x_var)
            x_out = x_mean, x_var
        else:
            x_out = self.flatten_layer(x)
        return x_out
