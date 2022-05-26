import tensorflow as tf

from .WeightType import WeightType


class QuantizedWeightsStraightThrough(WeightType):

    def __init__(self,
                 quantizer,
                 initializer=None,
                 regularize_l1=0.0,
                 regularize_l2=0.0,
                 w_constraints=(-1.0, 1.0)):
        super(WeightType, self).__init__()
        assert regularize_l1 >= 0.0
        assert regularize_l2 >= 0.0
        assert w_constraints is None or (isinstance(w_constraints, tuple) and len(w_constraints) == 2)
        self.quantizer = quantizer
        self.initializer = initializer if initializer is not None else 'glorot_uniform'
        self.regularize_l1 = regularize_l1
        self.regularize_l2 = regularize_l2
        w_constraints = (None, None) if w_constraints is None else w_constraints
        w_constraints = (None if w_constraints[0] == float('-inf') else w_constraints[0],
                         None if w_constraints[1] == float('+inf') else w_constraints[1])
        self.w_constraints = w_constraints
        self.w = None
        self.shape = None


    def initialize_weights(self, shape):
        self.shape = shape
        # Compute fan-in and fan-out according to tf implementation at
        # https://github.com/tensorflow/tensorflow/blob/r2.1/tensorflow/python/ops/init_ops.py#L1401 [retrieved on 10. Jan. 2020]
        # Note: In the following tutorial also the the pooling size is taken into account:
        # http://deeplearning.net/tutorial/lenet.html [retrieved on 10. Jan. 2020]
        if len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) == 4:
            receptive_field_size = shape[0] * shape[1]
            fan_in = receptive_field_size * shape[2]
            fan_out = receptive_field_size * shape[3]
        else:
            raise NotImplementedError('Shape with {} dimensions not supported'.format(len(shape)))

        if self.initializer == 'truncated_normal_sign':
            # Initializes to values in [-1,1]. Samples from a normal distribution and resamples values if they are not
            # within [-1,1]. This intialization is a safe choice when the quantizer has a quantization level at zero.
            w_init = tf.random.truncated_normal(shape)
        elif self.initializer == 'uniform_sign':
            # Initializes uniformly in [-1,1]. This intialization is a safe choice when the quantizer has a quantization
            # level at zero.
            w_init = tf.random.uniform(shape, minval=-1.0, maxval=1.0)
        elif self.initializer == 'glorot_uniform':
            # Glorot uniform. This initialization is unsafe when the quantizer has a quantization level at zero. In this
            # case, it can happen that all weights get quantized to zero and we do not learn anything.
            r = (6.0 / (fan_in + fan_out)) ** 0.5
            w_init = tf.random.uniform(shape, minval=-r, maxval=r)
        else:
            raise NotImplementedError('Initializer \'{}\' not implemented'.format(self.initializer))

        if self.w_constraints[0] is None and self.w_constraints[1] is None:
            constraint_fun = None
        elif self.w_constraints[0] is None:
            constraint_fun = lambda w : tf.minimum(w, self.w_constraints[1])
        elif self.w_constraints[1] is None:
            constraint_fun = lambda w : tf.maximum(w, self.w_constraints[0])
        else:
            assert self.w_constraints[0] < self.w_constraints[1]
            constraint_fun = lambda w : tf.clip_by_value(w, self.w_constraints[0], self.w_constraints[1])

        self.w = tf.Variable(w_init, trainable=True, name='RealWeights', constraint=constraint_fun)


    def apply_losses(self):
        if self.regularize_l1 > 0.0:
            self.add_loss(tf.reduce_sum(tf.abs(self.w)) * self.regularize_l1)
        if self.regularize_l2 > 0.0:
            self.add_loss(tf.reduce_sum(self.w ** 2.0) * self.regularize_l2)


    def is_deterministic(self):
        return True


    def expectation(self):
        return self.quantizer(self.w)


    def variance(self):
        return tf.zeros(self.shape, dtype=self.w.dtype)


    def expectation_and_variance(self):
        return self.expectation(), self.variance()

    
    def most_probable(self):
        return self.expectation()
