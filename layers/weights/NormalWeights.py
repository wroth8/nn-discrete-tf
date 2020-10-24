import tensorflow as tf
import math
import numpy as np

# from WeightType import WeightType
from layers.weights.WeightType import WeightType # TODO: Check why we have to specify the whole path and cannot leave out 'layers.weights.'

class NormalWeights(WeightType):

    def __init__(self,
                 regularize_kl=0.0,
                 regularize_kl_var=1e-2):
        super(WeightType, self).__init__()
        assert regularize_kl >= 0.0
        assert regularize_kl_var >= 0.0
        self.regularize_kl = regularize_kl
        self.regularize_kl_var = regularize_kl_var
        self.w_mean = None
        self.w_var_rho = None
        self.shape = None


    def initialize_weights(self, shape, initializer_mean='glorot_uniform', initializer_var='const'):
        self.shape = shape
        if isinstance(initializer_mean, WeightType):
            print('DEBUG::NormalWeights: Initialize with WeightType:expectation()')
            self.w_mean = tf.Variable(initializer_mean.expectation())
            assert self.w_mean.shape == shape
        elif initializer_mean == 'glorot_uniform':
            # glorot uniform
            r = (6.0 / (shape[0] + shape[1])) ** 0.5
            self.w_mean = tf.Variable(tf.random.uniform(shape, minval=-r, maxval=r))
        else:
            raise NotImplementedError('Initializer \'{}\' not implemented'.format(initializer_mean))

        if initializer_var == 'const':
            self.w_var_rho = tf.Variable(tf.constant(tf.math.log(1e-4), shape=shape),
                                         constraint=lambda w: tf.maximum(-10.0, w))
        else:
            raise NotImplementedError('Initializer \'{}\' not implemented'.format(initializer_var))


    def apply_losses(self):
        if self.regularize_kl > 0.0:
            w_var = self.variance()
            N = np.prod(self.shape)
            INV_VAR = 1.0 / self.regularize_kl_var
            LOG_VAR = math.log(self.regularize_kl_var)
            loss_kl = 0.5 * (LOG_VAR * N - N - tf.reduce_sum(tf.math.log(w_var)) + INV_VAR * (tf.reduce_sum(self.w_mean ** 2.0) + tf.reduce_sum(w_var)))
            self.add_loss(self.regularize_kl * loss_kl)


    def is_deterministic(self):
        return False


    def expectation(self):
        return self.w_mean


    def variance(self):
        return tf.nn.softplus(self.w_var_rho)


    def expectation_and_variance(self):
        return self.expectation(), self.variance()

    
    def most_probable(self):
        return self.expectation()

