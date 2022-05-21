import tensorflow as tf
import math


def tanhOfGaussian(m, v, epsilon=1e-6):
    # Gaussian approximation of the tanh applied to a Gaussian. Our implementation adapts ideas from
    # S. I. Wang, C. D. Manning; Fast dropout training; ICML 2013
    tanhSq_a = 4.0 - 2.0 * (2.0 ** 0.5)
    tanhSq_b = -math.log(2.0 ** 0.5 - 1.0)
    
    coef1 = (0.5 * math.pi) * v
    m_out = tf.math.tanh(m * tf.math.rsqrt(1.0 + coef1))
    v_out = 2.0 * (tf.math.tanh((m - tanhSq_b * 0.5) * tf.math.rsqrt(1.0 / tanhSq_a ** 2.0 + coef1)) + 1.0) \
            - tf.sqare(tf.math.tanh(m * tf.math.rsqrt(1.0 + coef1)) + 1.0) + epsilon
    return m_out, v_out


class DistTanh(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-6):
        super(DistTanh, self).__init__()
        self.epsilon = 1e-6


    def call(self, x, training):
        if training:
            if isinstance(x, tuple):
                assert len(x) == 2
                return self.call_train_distribution(x[0], x[1])
            else:
                return self.call_train_deterministic(x)
        else:
            assert not isinstance(x, tuple)
            return self.call_predict(x)


    def call_train_deterministic(self, x_in):
        return tf.math.tanh(x_in)


    def call_train_distribution(self, x_in_mean, x_in_var):
        return tanhOfGaussian(x_in_mean, x_in_var, epsilon=self.epsilon)


    def call_predict(self, x_in):
        return tf.math.tanh(x_in)
