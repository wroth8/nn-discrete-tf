import tensorflow as tf
import math
import numpy as np

# from WeightType import WeightType
from layers.weights.WeightType import WeightType # TODO: Check why we have to specify the whole path and cannot leave out 'layers.weights.'
from layers.weights.initializers import initialize_probabilities_from_expectation, map_to_ecdf

class QuinaryWeights(WeightType):

    def __init__(self,
                 regularize_shayer=0.0,
                 q_logit_constraints=(float('-inf'), float('+inf'))):

        super(WeightType, self).__init__()
        assert regularize_shayer >= 0.0
        assert q_logit_constraints is None or (isinstance(q_logit_constraints, tuple) and len(q_logit_constraints) == 2)
        self.regularize_shayer = regularize_shayer
        q_logit_constraints = (None, None) if q_logit_constraints is None else q_logit_constraints
        q_logit_constraints = (None if q_logit_constraints[0] == float('-inf') else q_logit_constraints[0],
                               None if q_logit_constraints[1] == float('+inf') else q_logit_constraints[1])
        self.q_logit_constraints = q_logit_constraints
        self.q_logits = None
        self.shape = None


    def initialize_weights(self, shape, initializer_logits='uniform'):
        self.shape = shape + (5,)
        if isinstance(initializer_logits, WeightType):
            w_expect = initializer_logits.expectation().numpy()
            # Use the empirical cdf to `stretch` the expected values
            idx_neg = w_expect <= 0.0
            idx_pos = np.logical_not(idx_neg)
            w_expect[idx_neg] = map_to_ecdf(w_expect[idx_neg]) * 1.25 - 1.25
            w_expect[idx_pos] = map_to_ecdf(w_expect[idx_pos]) * 1.25
            q_values = initialize_probabilities_from_expectation(w_expect, [-1.0, -0.5, 0.0, 0.5, 1.0])
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

        self.q_logits = tf.Variable(np.log(q_values), trainable=True, name='QuinaryWeightLogits', constraint=constraint_fun)
        if self.q_logits.shape != self.shape:
            raise Exception('Incorrect shapes: self.q_logits.shape={}, self.shape={}'.format(
                    self.q_logits.shape, self.shape))


    def apply_losses(self):
        if self.regularize_shayer > 0.0:
            self.add_loss(tf.reduce_sum(self.q_logits ** 2.0) * self.regularize_shayer)


    def is_deterministic(self):
        return False


    def probabilities(self):
        return tf.nn.softmax(self.q_logits, axis=-1)


    def expectation(self):
        q = self.probabilities()
        w_mean = -q[..., 0] + (q[..., 3] - q[..., 1]) * 0.5 + q[..., 4]
        return w_mean


    def variance(self):
        q = self.probabilities()
        w_mean = -q[..., 0] + (q[..., 3] - q[..., 1]) * 0.5 + q[..., 4]
        w_var = (q[..., 0] * tf.square(1.0 + w_mean) +
                 q[..., 1] * tf.square(0.5 + w_mean) +
                 q[..., 2] * tf.square(w_mean) +
                 q[..., 3] * tf.square(0.5 - w_mean) +
                 q[..., 4] * tf.square(1.0 - w_mean))
        return w_var


    def expectation_and_variance(self):
        q = self.probabilities()
        w_mean = -q[..., 0] + (q[..., 3] - q[..., 1]) * 0.5 + q[..., 4]
        w_var = (q[..., 0] * tf.square(1.0 + w_mean) +
                 q[..., 1] * tf.square(0.5 + w_mean) +
                 q[..., 2] * tf.square(w_mean) +
                 q[..., 3] * tf.square(0.5 - w_mean) +
                 q[..., 4] * tf.square(1.0 - w_mean))
        return w_mean, w_var


    def most_probable(self):
        w_mp = (tf.cast(tf.math.argmax(self.q_logits, axis=-1), self.q_logits.dtype) - 2.0) * 0.5
        return w_mp
