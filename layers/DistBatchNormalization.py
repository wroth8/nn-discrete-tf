import tensorflow as tf

from tensorflow.python.ops.state_ops import assign
from tensorflow.python.ops.state_ops import assign_sub
from tensorflow.python.ops.nn_impl import fused_batch_norm


class DistBatchNormalization(tf.keras.layers.Layer):

    def __init__(self,
                 momentum=0.99,
                 epsilon=1e-6):
        super(DistBatchNormalization, self).__init__()
        assert 0.0 < momentum and momentum < 1.0
        self.momentum = momentum
        self.decay = 1.0 - self.momentum
        self.epsilon = epsilon
        self.beta = None
        self.gamma = None
        self.ema_batch_mean = None
        self.ema_batch_inv_std = None


    def build(self, input_shape):
        if isinstance(input_shape, tuple):
            assert len(input_shape) == 2
            num_channels = input_shape[0][-1]
        else:
            num_channels = input_shape[-1]

        self.beta = self.add_weight(
                name='DistBatchNormalization_beta',
                shape=(num_channels,),
                initializer='zeros',
                trainable=True)
        self.gamma = self.add_weight(
                name='DistBatchNormalization_gamma',
                shape=(num_channels,),
                initializer='ones',
                trainable=True)
        self.ema_batch_mean = self.add_weight(
                name='DistBatchNormalization_ema_batch_mean',
                shape=(num_channels,),
                initializer='zeros',
                trainable=False)
        self.ema_batch_inv_std = self.add_weight(
                name='DistBatchNormalization_ema_batch_inv_std',
                shape=(num_channels,),
                initializer='ones',
                trainable=False)


    def call(self, x, training, enable_ema_updates=None):
        if enable_ema_updates is None:
            enable_ema_updates = training
        if training:
            if isinstance(x, tuple):
                assert len(x) == 2
                return self.call_train_distribution(x[0], x[1], enable_ema_updates)
            else:
                return self.call_train_deterministic(x, enable_ema_updates)
        else:
            assert enable_ema_updates == False
            return self.call_predict(x)

    
    def call_train_deterministic(self, x_in, enable_ema_updates):
        if x_in.shape.rank == 4:
            # Fused batch norm is way faster than our own implementation
            x_out, batch_mean, batch_var = fused_batch_norm(
                    x_in,
                    self.gamma,
                    self.beta,
                    epsilon=self.epsilon,
                    data_format='NHWC',
                    is_training=True)
            batch_inv_std = tf.math.rsqrt(batch_var + self.epsilon)
        else:
            reduce_dims = list(range(x_in.shape.rank - 1)) # reduce all but the last dimension
            n_reduce_inv = 1.0 / (tf.cast(tf.reduce_prod(x_in.shape[:-1]), tf.float32) - 1.0)
            
            batch_mean = tf.reduce_mean(x_in, axis=reduce_dims)
            x_centered = x_in - batch_mean
            batch_var = tf.reduce_sum(tf.square(x_centered), axis=reduce_dims) * n_reduce_inv
            batch_inv_std = tf.math.rsqrt(batch_var + self.epsilon)
            scaling_factor = batch_inv_std * self.gamma
            x_scaled = x_centered * scaling_factor
            x_out = x_scaled + self.beta
        
        if enable_ema_updates:
            self.add_update(assign_sub(self.ema_batch_mean, self.decay * (self.ema_batch_mean - batch_mean)))
            self.add_update(assign_sub(self.ema_batch_inv_std, self.decay * (self.ema_batch_inv_std - batch_inv_std)))
            # The following code is equivalent
            # self.add_update(assign(self.ema_batch_mean, self.momentum * self.ema_batch_mean + (1.0 - self.momentum) * batch_mean))
            # self.add_update(assign(self.ema_batch_inv_std, self.momentum * self.ema_batch_inv_std + (1.0 - self.momentum) * batch_inv_std))
                
        return x_out


    def call_train_distribution(self, x_in_mean, x_in_var, enable_ema_updates):
        reduce_dims = list(range(x_in_mean.shape.rank - 1)) # reduce all but the last dimension
        n_reduce_inv = 1.0 / (tf.cast(tf.reduce_prod(x_in_mean.shape[:-1]), tf.float32) - 1.0)
        batch_mean = tf.reduce_mean(x_in_mean, axis=reduce_dims)
        batch_var = tf.reduce_sum(x_in_var + tf.square(x_in_mean - batch_mean), axis=reduce_dims) * n_reduce_inv
        batch_inv_std = tf.math.rsqrt(batch_var + self.epsilon)

        if enable_ema_updates:
            self.add_update(assign_sub(self.ema_batch_mean, self.decay * (self.ema_batch_mean - batch_mean)))
            self.add_update(assign_sub(self.ema_batch_inv_std, self.decay * (self.ema_batch_inv_std - batch_inv_std)))
            # The following code is equivalent
            # self.add_update(assign(self.ema_batch_mean, self.momentum * self.ema_batch_mean + (1.0 - self.momentum) * batch_mean))
            # self.add_update(assign(self.ema_batch_inv_std, self.momentum * self.ema_batch_inv_std + (1.0 - self.momentum) * batch_inv_std))

        a = batch_inv_std * self.gamma
        b = self.beta - a * batch_mean
        x_out_mean = x_in_mean * a + b
        x_out_var = x_in_var * tf.square(a)
        return x_out_mean, x_out_var


    def call_predict(self, x_in):
        a = self.ema_batch_inv_std * self.gamma
        b = self.beta - self.ema_batch_inv_std * self.ema_batch_mean * self.gamma
        return x_in * a + b
