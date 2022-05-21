import tensorflow as tf

# from WeightType import WeightType
from layers.weights.WeightType import WeightType # TODO: Check why we have to specify the whole path and cannot leave out 'layers.weights.'


class RealWeights(WeightType):

    def __init__(self,
                 initializer=None,
                 regularize_l1=0.0,
                 regularize_l2=0.0):
        super(WeightType, self).__init__()
        assert regularize_l1 >= 0.0
        assert regularize_l2 >= 0.0
        self.initializer = initializer if initializer is not None else 'glorot_uniform'
        self.regularize_l1 = regularize_l1
        self.regularize_l2 = regularize_l2
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

        LAMBDA = lambda:0 # required to check if we got a lambda: https://stackoverflow.com/questions/3655842/how-can-i-test-whether-a-variable-holds-a-lambda
        if self.initializer == 'glorot_uniform':
            # Glorot uniform
            r = (6.0 / (fan_in + fan_out)) ** 0.5
            self.w = tf.Variable(tf.random.uniform(shape, minval=-r, maxval=r), trainable=True, name='RealWeights')
        elif isinstance(self.initializer, type(LAMBDA)) and self.initializer.__name__ == LAMBDA.__name__:
            # We expect a lambda that takes inputs (shape,fan_in,fan_out) and creates a tf tensor of the given shape
            self.w = self.initializer(shape, fan_in, fan_out)
        else:
            raise NotImplementedError('Initializer \'{}\' not implemented'.format(self.initializer))


    def apply_losses(self):
        if self.regularize_l1 > 0.0:
            self.add_loss(tf.reduce_sum(tf.abs(self.w)) * self.regularize_l1)
        if self.regularize_l2 > 0.0:
            self.add_loss(tf.reduce_sum(self.w ** 2.0) * self.regularize_l2)


    def is_deterministic(self):
        return True


    def expectation(self):
        return self.w


    def variance(self):
        return tf.zeros(self.shape, dtype=self.w.dtype)


    def expectation_and_variance(self):
        return self.expectation(), self.variance()

    
    def most_probable(self):
        return self.w
