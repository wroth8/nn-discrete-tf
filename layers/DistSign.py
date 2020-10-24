import tensorflow as tf
import math

# TODO: see if we can remove the full path 'layers.ste'
from layers.ste import sign0_ste_id, sign0_ste_tanh, sign0_ste_hardtanh, \
                       sign_ste_id, sign_ste_tanh, sign_ste_hardtanh, \
                       sign_stoch_ste_id, sign_stoch_ste_tanh, sign_stoch_ste_hardtanh


def signOfGaussian(m, v, epsilon=1e-6):
    # Gaussian approximation of the sign applied to a Gaussian
    m_out = tf.math.erf(m * tf.math.rsqrt(2.0 * v + epsilon))
    v_out = 1.0 - tf.square(m_out) + max(1e-6, epsilon) # the max prevented 1.0 + eps being reduced to 1.0
    return m_out, v_out


class DistSign(tf.keras.layers.Layer):

    def __init__(self, has_zero_output=True, straight_through_type=None, stochastic=False, epsilon=1e-6):
        super(DistSign, self).__init__()
        assert straight_through_type is None or straight_through_type in ['id', 'tanh', 'hardtanh']
        self.has_zero_output = has_zero_output
        self.straight_through_type = straight_through_type
        self.stochastic = stochastic
        self.epsilon = epsilon


    def call(self, x, training):
        if training:
            if isinstance(x, tuple):
                assert len(x) == 2
                return self.call_train_distribution(x[0], x[1])
            else:
                return self.call_train_deterministic(x)
        else:
            return self.call_predict(x)


    def call_train_deterministic(self, x_in):
        if self.straight_through_type is None:
            print('Warning: Applying sign function to deterministic inputs. Backpropagation will result in zero gradients')
            if self.has_zero_output:
                return tf.math.sign(x_in)
            else:
                return tf.cast(tf.where(x_in >= 0.0, 1.0, -1.0), x_in.dtype)
        else:
            if self.stochastic:
                if self.straight_through_type == 'id':
                    return sign_stoch_ste_id(x_in)
                elif self.straight_through_type == 'tanh':
                    return sign_stoch_ste_tanh(x_in)
                elif self.straight_through_type == 'hardtanh':
                    return sign_stoch_ste_hardtanh(x_in)
                else:
                    raise NotImplementedError('DistSign: Straight-through type \'{}\' not implemented'.format(self.straight_through_type))
            elif self.has_zero_output:
                if self.straight_through_type == 'id':
                    return sign0_ste_id(x_in)
                elif self.straight_through_type == 'tanh':
                    return sign0_ste_tanh(x_in)
                elif self.straight_through_type == 'hardtanh':
                    return sign0_ste_hardtanh(x_in)
                else:
                    raise NotImplementedError('DistSign: Straight-through type \'{}\' not implemented'.format(self.straight_through_type))
            else:
                if self.straight_through_type == 'id':
                    return sign_ste_id(x_in)
                elif self.straight_through_type == 'tanh':
                    return sign_ste_tanh(x_in)
                elif self.straight_through_type == 'hardtanh':
                    return sign_ste_hardtanh(x_in)
                else:
                    raise NotImplementedError('DistSign: Straight-through type \'{}\' not implemented'.format(self.straight_through_type))


    def call_train_distribution(self, x_in_mean, x_in_var):
        return signOfGaussian(x_in_mean, x_in_var, epsilon=self.epsilon)


    def call_predict(self, x_in):
        if self.has_zero_output:
            return tf.math.sign(x_in)
        else:
            return tf.cast(tf.where(x_in >= 0.0, 1.0, -1.0), x_in.dtype)
