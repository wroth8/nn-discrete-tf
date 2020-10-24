import tensorflow as tf
import math
import numpy as np

# from WeightType import WeightType
from layers.weights.WeightType import WeightType # TODO: Check why we have to specify the whole path and cannot leave out 'layers.weights.'
from layers.weights.initializers import initialize_shayer_probabilities_from_expectation

class TernaryWeightsShayer(WeightType):

    def __init__(self,
                 regularize_shayer=0.0,
                 enable_unsafe_variance=True):
        '''
        enable_unsafe_variance: If set to true, the variances are computed in a more efficient way, but we cannot
          guarantee that the resulting values will be non-negative (`catastrophic cancellation`). In particular, we used
          wolframalpha to simplify the expression for computing variances to minimize the number of mutliplications. We
          do not recommend to use this implementation unless the logits are bound so that the corresponding
          probabilities are not too close to zero or one.
          @see https://www.wolframalpha.com/input/?i=simplify+%281+%2B+x%29%5E2+*+%281+-+a%29+*+%281+-+b%29+%2B+x%5E2+*+a+%2B+%281+-+x%29%5E2+*+%281+-+a%29+*+b
        '''
        super(WeightType, self).__init__()
        assert regularize_shayer >= 0.0
        self.regularize_shayer = regularize_shayer
        self.enable_unsafe_variance = enable_unsafe_variance
        self.q_zro_logits = None
        self.q_cond_pos_logits = None # this is actually a conditional probabilities q(w=1|w!=0)
        self.shape = None


    def initialize_weights(self, shape, initializer_logits='uniform'):
        self.shape = shape
        if isinstance(initializer_logits, WeightType):
            w_expect = initializer_logits.expectation().numpy()
            w_expect = w_expect / np.std(w_expect)
            q_zro_values, q_cond_pos_values = initialize_shayer_probabilities_from_expectation(w_expect, [-1.0, 0.0, 1.0])
        elif initializer_logits == 'uniform':
            q_zro_values = tf.random.uniform(self.shape, minval=0.4, maxval=0.6)
            q_cond_pos_values = tf.random.uniform(self.shape, minval=0.4, maxval=0.6)
        else:
            raise NotImplementedError('Initializer \'{}\' not implemented'.format(initializer_logits))

        self.q_zro_logits = tf.Variable(
                np.log(q_zro_values / (1.0 - q_zro_values)), # logit function (inverse of logistic sigmoid function)
                trainable=True,
                name='TernaryShayerWeightZroLogits')
        self.q_cond_pos_logits = tf.Variable(
                np.log(q_cond_pos_values / (1.0 - q_cond_pos_values)),
                trainable=True,
                name='TernaryShayerWeightCondPosLogits')
        if self.q_zro_logits.shape != self.shape:
            raise Exception('Incorrect shapes: self.q_zro_logits.shape={}, self.shape={}'.format(
                    self.q_zro_logits.shape, self.shape))
        if self.q_cond_pos_logits.shape != self.shape:
            raise Exception('Incorrect shapes: self.q_cond_pos_logits.shape={}, self.shape={}'.format(
                    self.q_cond_pos_logits.shape, self.shape))

    def apply_losses(self):
        if self.regularize_shayer > 0.0:
            self.add_loss((tf.reduce_sum(self.q_zro_logits ** 2.0) +
                           tf.reduce_sum(self.q_cond_pos_logits ** 2.0)) * self.regularize_shayer)


    def is_deterministic(self):
        return False


    def probabilities(self):
        q_zro = tf.math.sigmoid(self.q_zro_logits)
        q_cond_pos = tf.math.sigmoid(self.q_cond_pos_logits)
        q_non_zro = (1.0 - q_zro)
        q_pos = q_non_zro * q_cond_pos
        q_neg = q_non_zro * (1.0 - q_cond_pos)
        return tf.stack([q_neg, q_zro, q_pos], axis=-1)


    def expectation(self):
        q_zro = tf.math.sigmoid(self.q_zro_logits)
        q_cond_pos = tf.math.sigmoid(self.q_cond_pos_logits)
        q_non_zro = 1.0 - q_zro
        w_mean = q_non_zro * (2.0 * q_cond_pos - 1.0)
        return w_mean


    def variance(self):
        q_zro = tf.math.sigmoid(self.q_zro_logits)
        q_cond_pos = tf.math.sigmoid(self.q_cond_pos_logits)
        q_non_zro = 1.0 - q_zro

        if self.enable_unsafe_variance:
            # According to wolframalpha we have
            # w_var = q_zro * ((4 * q_cond_pos - 2) * w_mean - 1) + w_mean * (w_mean - (4 * q_cond_pos - 2)) + 1
            coef1 = 2.0 * q_cond_pos - 1.0
            w_mean = q_non_zro * coef1
            coef2 = 2.0 * coef1 * w_mean - 1.0
            w_var = coef2 * (q_zro - 1.0) + tf.square(w_mean)
        else:
            w_mean = q_non_zro * (2.0 * q_cond_pos - 1.0)
            w_var = (tf.square(1.0 + w_mean) * q_non_zro * (1.0 - q_cond_pos) +
                     tf.square(w_mean) * q_zro + 
                     tf.square(1.0 - w_mean) * q_non_zro * q_cond_pos)
        return w_var


    def expectation_and_variance(self):
        q_zro = tf.math.sigmoid(self.q_zro_logits)
        q_cond_pos = tf.math.sigmoid(self.q_cond_pos_logits)
        q_non_zro = (1.0 - q_zro)
        
        if self.enable_unsafe_variance:
            # According to wolframalpha we have
            # w_var = q_zro * ((4 * q_cond_pos - 2) * w_mean - 1) + w_mean * (w_mean - (4 * q_cond_pos - 2)) + 1
            coef1 = 2.0 * q_cond_pos - 1.0
            w_mean = q_non_zro * coef1
            coef2 = 2.0 * coef1 * w_mean - 1.0
            w_var = coef2 * (q_zro - 1.0) + tf.square(w_mean)
        else:
            w_mean = q_non_zro * (2.0 * q_cond_pos - 1.0)
            w_var = (tf.square(1.0 + w_mean) * q_non_zro * (1.0 - q_cond_pos) +
                     tf.square(w_mean) * q_zro + 
                     tf.square(1.0 - w_mean) * q_non_zro * q_cond_pos)

        return w_mean, w_var


    def most_probable(self):
        w_probs = self.probabilities()
        w_mp = tf.cast(tf.math.argmax(w_probs, axis=-1), w_probs.dtype) - 1.0
        return w_mp

