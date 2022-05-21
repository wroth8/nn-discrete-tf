import tensorflow as tf


class DistDense(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 dist_weights,
                 use_bias=True,
                 enable_activation_normalization=False):
        super(DistDense, self).__init__()
        self.units_out = units
        self.units_in = None
        self.dist_weights = dist_weights
        self.use_bias = use_bias
        self.bias = None
        self.enable_activation_normalization = enable_activation_normalization


    def build(self, input_shape):
        if isinstance(input_shape, tuple):
            self.units_in = input_shape[0][-1]
        else:
            self.units_in = input_shape[-1]

        self.dist_weights.initialize_weights((self.units_in, self.units_out))

        if self.use_bias:
            self.bias = tf.Variable(tf.zeros(shape=(self.units_out,)))


    def call(self, x, training, use_sampled_weights=False):
        assert not (use_sampled_weights and training) # sampled weights can only be used for predictions
        if training:
            self.dist_weights.apply_losses() # Apply regularization
            if isinstance(x, tuple):
                assert len(x) == 2
                return self.call_train_distribution(x[0], x[1])
            else:
                return self.call_train_deterministic(x)
        else:
            return self.call_predict(x, use_sampled_weights=use_sampled_weights)


    def call_train_deterministic(self, x_in):
        if self.dist_weights.is_deterministic():
            w_mean = self.dist_weights.expectation()
            x_out = tf.matmul(x_in, w_mean)
            if self.enable_activation_normalization:
                x_out = x_out * (float(self.units_in) ** -0.5)
            if self.use_bias:
                x_out = x_out + self.bias
            return x_out
        else:
            w_mean, w_var = self.dist_weights.expectation_and_variance()
            x_out_mean = tf.matmul(x_in, w_mean)
            x_out_var = tf.matmul(x_in ** 2.0, w_var)
            if self.enable_activation_normalization:
                x_out_mean = x_out_mean * (float(self.units_in) ** -0.5)
                x_out_var = x_out_var * (float(self.units_in) ** -1.0)
            if self.use_bias:
                x_out_mean = x_out_mean + self.bias
            return x_out_mean, x_out_var


    def call_train_distribution(self, x_in_mean, x_in_var):
        if self.dist_weights.is_deterministic():
            w_mean = self.dist_weights.expectation()
            x_out_mean = tf.matmul(x_in_mean, w_mean)
            x_out_var = tf.matmul(x_in_var, w_mean ** 2.0)
        else:
            w_mean, w_var = self.dist_weights.expectation_and_variance()
            x_out_mean = tf.matmul(x_in_mean, w_mean)
            x_out_var = tf.matmul(x_in_mean ** 2.0, w_var) + \
                        tf.matmul(x_in_var, w_mean ** 2.0) + \
                        tf.matmul(x_in_var, w_var)
        if self.enable_activation_normalization:
            x_out_mean = x_out_mean * (float(self.units_in) ** -0.5)
            x_out_var = x_out_var * (float(self.units_in) ** -1.0)
        if self.use_bias:
            x_out_mean = x_out_mean + self.bias
        return x_out_mean, x_out_var


    def call_predict(self, x_in, use_sampled_weights=False):
        if use_sampled_weights:
            w = self.dist_weights.sampled()
        else:
            w = self.dist_weights.most_probable()
        x_out = tf.matmul(x_in, w)
        if self.enable_activation_normalization:
            x_out = x_out * (float(self.units_in) ** -0.5)
        if self.use_bias:
            x_out = x_out + self.bias
        return x_out


    def resample_weights(self):
        self.dist_weights.resample_weights()