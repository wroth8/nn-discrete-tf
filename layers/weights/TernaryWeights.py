import numpy as np
import tensorflow as tf

from .initializers import initialize_probabilities_from_expectation, initialize_shayer_probabilities_from_expectation, map_to_ecdf
from .WeightType import WeightType


class TernaryWeights(WeightType):

    def __init__(self,
                 regularize_shayer=0.0,
                 enable_sampled_weights=False,
                 enable_unsafe_variance=False,
                 q_logit_constraints=(float('-inf'), float('+inf')),
                 initializer_mode='default'):
        '''
        enable_unsafe_variance: If set to true, the variances are computed in a more efficient way, but we cannot
          guarantee that the resulting values will be non-negative (`catastrophic cancellation`). In particular, we used
          wolframalpha to simplify the expression for computing variances. We do not recommend to use this implementation
          unless the logits are bound so that the corresponding probabilities are not too close to zero or one.
          @see https://www.wolframalpha.com/input/?i=simplify+%281+%2B+c+-+a%29%5E2+*+a+%2B+%28c+-+a%29%5E2+*+%281+-+a+-+c%29+%2B+%281+-+c+%2B+a%29%5E2+*+c
        '''
        super(WeightType, self).__init__()
        assert regularize_shayer >= 0.0
        assert q_logit_constraints is None or (isinstance(q_logit_constraints, tuple) and len(q_logit_constraints) == 2)
        self.regularize_shayer = regularize_shayer
        self.enable_sampled_weights = enable_sampled_weights # setting this to False saves the memory for the sampled weights
        self.enable_unsafe_variance = enable_unsafe_variance
        q_logit_constraints = (None, None) if q_logit_constraints is None else q_logit_constraints
        q_logit_constraints = (None if q_logit_constraints[0] == float('-inf') else q_logit_constraints[0],
                               None if q_logit_constraints[1] == float('+inf') else q_logit_constraints[1])
        self.q_logit_constraints = q_logit_constraints
        if initializer_mode is None:
            initializer_mode = 'default'
        assert initializer_mode in ['default', 'roth', 'roth_without_normalization', 'shayer', 'shayer_without_normalization']
        self.initializer_mode = initializer_mode
        self.q_logits = None
        self.w_sampled = None
        self.shape = None


    def initialize_weights(self, shape, initializer_logits='uniform'):
        self.shape = shape + (3,)
        if isinstance(initializer_logits, WeightType):
            if self.initializer_mode in ['default', 'roth']:
                w_expect = initializer_logits.expectation().numpy()
                # Use the empirical cdf to `stretch` the expected values
                idx_neg = w_expect <= 0.0
                idx_pos = np.logical_not(idx_neg)
                w_expect[idx_neg] = map_to_ecdf(w_expect[idx_neg]) * 1.5 - 1.5
                w_expect[idx_pos] = map_to_ecdf(w_expect[idx_pos]) * 1.5
                q_values = initialize_probabilities_from_expectation(w_expect, [-1.0, 0.0, 1.0])
            elif self.initializer_mode == 'roth_without_normalization':
                w_expect = initializer_logits.expectation().numpy()
                q_values = initialize_probabilities_from_expectation(w_expect, [-1.0, 0.0, 1.0])
            elif self.initializer_mode == 'shayer':
                w_expect = initializer_logits.expectation().numpy()
                w_expect = w_expect / np.std(w_expect)
                q_zro_values, q_cond_pos_values = initialize_shayer_probabilities_from_expectation(w_expect, [-1.0, 0.0, 1.0])
                q_values = np.stack([(1.0 - q_zro_values) * (1.0 - q_cond_pos_values),
                                     q_zro_values,
                                     (1.0 - q_zro_values) * q_cond_pos_values], axis=-1)
            elif self.initializer_mode == 'shayer_without_normalization':
                w_expect = initializer_logits.expectation().numpy()
                q_zro_values, q_cond_pos_values = initialize_shayer_probabilities_from_expectation(w_expect, [-1.0, 0.0, 1.0])
                q_values = np.stack([(1.0 - q_zro_values) * (1.0 - q_cond_pos_values),
                                     q_zro_values,
                                     (1.0 - q_zro_values) * q_cond_pos_values], axis=-1)
            else:
                raise NotImplementedError('Initializer mode \'{}\' not implemented'.format(self.initializer_mode))
        elif initializer_logits == 'uniform':
            q_values = tf.random.uniform(self.shape, minval=-1.0, maxval=1.0)
            q_values = tf.math.softmax(q_values, axis=-1)
        else:
            raise NotImplementedError('Initializer \'{}\' not implemented'.format(initializer_logits))

        if self.q_logit_constraints[0] is None and self.q_logit_constraints[1] is None:
            constraint_fun = None
        elif self.q_logit_constraints[0] is None:
            constraint_fun = lambda w : tf.minimum(w, self.q_logit_constraints[1])
        elif self.q_logit_constraints[1] is None:
            constraint_fun = lambda w : tf.maximum(w, self.q_logit_constraints[0])
        else:
            assert self.q_logit_constraints[0] < self.q_logit_constraints[1]
            constraint_fun = lambda w : tf.clip_by_value(w, self.q_logit_constraints[0], self.q_logit_constraints[1])

        self.q_logits = tf.Variable(np.log(q_values), trainable=True, name='TernaryWeightLogits', constraint=constraint_fun)
        if self.q_logits.shape != self.shape:
            raise Exception('Incorrect shapes: self.q_logits.shape={}, self.shape={}'.format(
                    self.q_logits.shape, self.shape))

        if self.enable_sampled_weights:
            self.w_sampled = tf.Variable(np.zeros(self.shape[:-1]), trainable=False, name='TernaryWeightsSampled', dtype=self.q_logits.dtype)


    def apply_losses(self):
        if self.regularize_shayer > 0.0:
            self.add_loss(tf.reduce_sum(self.q_logits ** 2.0) * self.regularize_shayer)


    def is_deterministic(self):
        return False


    def probabilities(self):
        return tf.nn.softmax(self.q_logits, axis=-1)


    def expectation(self):
        q = self.probabilities()
        w_mean = -q[..., 0] + q[..., 2]
        return w_mean


    def variance(self):
        q = self.probabilities()
        w_mean = -q[..., 0] + q[..., 2]
        if self.enable_unsafe_variance:
            w_var = -tf.square(w_mean) + q[..., 0] + q[..., 2]
        else:
            w_var = (tf.square(1.0 + w_mean) * q[..., 0] +
                     tf.square(w_mean) * q[..., 1] + 
                     tf.square(1.0 - w_mean) * q[..., 2])
        return w_var


    def expectation_and_variance(self):
        q = self.probabilities()
        w_mean = -q[..., 0] + q[..., 2]
        if self.enable_unsafe_variance:
            w_var = -tf.square(w_mean) + q[..., 0] + q[..., 2]
        else:
            w_var = (tf.square(1.0 + w_mean) * q[..., 0] +
                     tf.square(w_mean) * q[..., 1] + 
                     tf.square(1.0 - w_mean) * q[..., 2])
        return w_mean, w_var


    def most_probable(self):
        w_mp = tf.cast(tf.math.argmax(self.q_logits, axis=-1), self.q_logits.dtype) - 1.0
        return w_mp


    def resample_weights(self):
        assert self.enable_sampled_weights
        assert self.w_sampled is not None # model not initialized?
        w_resampled = tf.cast(tf.random.categorical(tf.reshape(self.q_logits, [-1, 3]), 1), self.q_logits.dtype) - 1.0
        w_resampled = tf.reshape(w_resampled, self.shape[:-1])
        self.w_sampled.assign(w_resampled)


    def sampled(self):
        assert self.enable_sampled_weights
        assert self.w_sampled is not None # model not initialized?
        return self.w_sampled