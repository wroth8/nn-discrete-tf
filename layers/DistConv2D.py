import tensorflow as tf


class DistConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 dist_weights,
                 use_bias=True,
                 enable_activation_normalization=False,
                 data_format='NHWC'):
        super(DistConv2D, self).__init__()
        self.channels_out = filters
        self.channels_in = None
        self.kernel_size = kernel_size
        self.dist_weights = dist_weights
        self.use_bias = use_bias
        self.strides = (1, 1)
        self.padding = 'SAME'
        self.enable_activation_normalization = enable_activation_normalization
        self.data_format = data_format


    def build(self, input_shape):
        if self.data_format == 'NHWC':
            if isinstance(input_shape, tuple):
                self.channels_in = input_shape[0][-1]
            else:
                self.channels_in = input_shape[-1]
            self.dist_weights.initialize_weights((self.kernel_size[0],
                                                  self.kernel_size[1],
                                                  self.channels_in,
                                                  self.channels_out))
            if self.use_bias:
                self.bias = tf.Variable(tf.zeros(shape=(self.channels_out,)))
        else:
            raise NotImplementedError('DistConv2D: Data format \'{}\' not implemented'.format(data_format))


    def call(self, x, training):
        if training:
            self.dist_weights.apply_losses() # Apply regularization
            if isinstance(x, tuple):
                assert len(x) == 2
                return self.call_train_distribution(x[0], x[1])
            else:
                return self.call_train_deterministic(x)
        else:
            return self.call_predict(x)


    def call_train_deterministic(self, x_in):
        if self.dist_weights.is_deterministic():
            w_mean = self.dist_weights.expectation()
            x_out = tf.nn.conv2d(x_in, w_mean, self.strides, self.padding, data_format=self.data_format)
            if self.enable_activation_normalization:
                x_out = x_out * (float(self.channels_in * self.kernel_size[0] * self.kernel_size[1]) ** -0.5)
            if self.use_bias:
                x_out = x_out + self.bias
            return x_out
        else:
            w_mean, w_var = self.dist_weights.expectation_and_variance()
            x_out_mean = tf.nn.conv2d(x_in, w_mean, self.strides, self.padding, data_format=self.data_format)
            x_out_var = tf.nn.conv2d(x_in ** 2.0, w_var, self.strides, self.padding, data_format=self.data_format)
            if self.enable_activation_normalization:
                x_out_mean = x_out_mean * (float(self.channels_in * self.kernel_size[0] * self.kernel_size[1]) ** -0.5)
                x_out_var = x_out_var * (float(self.channels_in * self.kernel_size[0] * self.kernel_size[1]) ** -1.0)
            if self.use_bias:
                x_out_mean = x_out_mean + self.bias
            return x_out_mean, x_out_var


    def call_train_distribution(self, x_in_mean, x_in_var):
        if self.dist_weights.is_deterministic():
            w_mean = self.dist_weights.expectation()
            x_out_mean = tf.nn.conv2d(x_in_mean, w_mean, self.strides, self.padding, data_format=self.data_format)
            x_out_var = tf.nn.conv2d(x_in_var, w_mean ** 2.0, self.strides, self.padding, data_format=self.data_format)
        else:
            w_mean, w_var = self.dist_weights.expectation_and_variance()
            x_out_mean = tf.nn.conv2d(x_in_mean, w_mean, self.strides, self.padding, data_format=self.data_format)
            x_out_var = tf.nn.conv2d(x_in_mean ** 2.0, w_var, self.strides, self.padding, data_format=self.data_format) + \
                        tf.nn.conv2d(x_in_var, w_mean ** 2.0, self.strides, self.padding, data_format=self.data_format) + \
                        tf.nn.conv2d(x_in_var, w_var, self.strides, self.padding, data_format=self.data_format)
        if self.enable_activation_normalization:
            x_out_mean = x_out_mean * (float(self.channels_in * self.kernel_size[0] * self.kernel_size[1]) ** -0.5)
            x_out_var = x_out_var * (float(self.channels_in * self.kernel_size[0] * self.kernel_size[1]) ** -1.0)
        if self.use_bias:
            x_out_mean = x_out_mean + self.bias
        return x_out_mean, x_out_var


    def call_predict(self, x_in):
        w_mp = self.dist_weights.most_probable()
        x_out = tf.nn.conv2d(x_in, w_mp, self.strides, self.padding, data_format=self.data_format)
        if self.enable_activation_normalization:
            x_out = x_out * (float(self.channels_in * self.kernel_size[0] * self.kernel_size[1]) ** -0.5)
        if self.use_bias:
            x_out = x_out + self.bias
        return x_out
