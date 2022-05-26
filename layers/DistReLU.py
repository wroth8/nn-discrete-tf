import math
import tensorflow as tf


def reluOfGaussian(m, v, epsilon=1e-6):
    # Gaussian approximation of the relu applied to a Gaussian
    # Implementation according to:
    # Hernandez-Lobato and Adams; Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks
    ALPHA_MIN = -5.0 # alpha values less than ALPHA_MIN result in numerical issues (see Hernandez-Lobato)
    alpha = m * tf.math.rsqrt(v + epsilon)

    # The following implementation seems to be faster
    pdf_norm = float(1.0 / (2.0 * math.pi) ** 0.5)        
    pdf_alpha = pdf_norm * tf.exp(-0.5 * alpha ** 2.0)
    alpha_div_sqrt2 = alpha * (0.5 ** 0.5)
    cdf_alpha_pos = 0.5 * (1.0 + tf.math.erf(alpha_div_sqrt2))
    cdf_alpha_neg = 1.0 - cdf_alpha_pos
    # cdf_alpha_neg = 0.5 * (1.0 + tf.math.erf(-alpha_div_sqrt2)) # TODO: try with 1. - cdf_alpha_pos and see if it improves performance

    # Note: The tf.maximum is important here to avoid 0 / 0. Even if the following tf.where does not select this
    # computation path, it will cause NaN-gradients because the gradient of the path that is not selected is just
    # multiplied by 0 which still results in NaN gradients.
    # @see https://stackoverflow.com/questions/50187342/tensorflow-gradient-with-tf-where-returns-nan-when-it-shouldnt
    gamma1 = pdf_alpha / tf.maximum(cdf_alpha_pos, 1e-10)
    # Note: We have to take care of the other path of tf.where too for the same reason. If alpha equals zero, we
    # have a division by zero and the gradient will also be NaN (it really happens).
    alpha_inv = 1.0 / tf.minimum(alpha, ALPHA_MIN) # For the gamma2-path we have alpha <= ALPHA_MIN
    gamma2 = -alpha - alpha_inv + 2.0 * alpha_inv ** 3.0

    gamma = tf.where(alpha >= ALPHA_MIN, gamma1, gamma2)

    v_aux = m + tf.sqrt(v + epsilon) * gamma
    m_out = cdf_alpha_pos * v_aux
    v_out = m_out * v_aux * cdf_alpha_neg + cdf_alpha_pos * v * (1. - gamma * (gamma + alpha))
    v_out = tf.maximum(v_out, epsilon)

    return m_out, v_out


class DistReLU(tf.keras.layers.Layer):

    def __init__(self, epsilon=1e-6):
        super(DistReLU, self).__init__()
        self.epsilon = epsilon


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
        return tf.nn.relu(x_in)


    def call_train_distribution(self, x_in_mean, x_in_var):
        return reluOfGaussian(x_in_mean, x_in_var, epsilon=self.epsilon)


    def call_predict(self, x_in):
        return tf.nn.relu(x_in)
