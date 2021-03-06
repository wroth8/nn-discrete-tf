import tensorflow as tf
import math


def maxOfGaussians(m1, v1, m2, v2, epsilon=1e-6):
    # Gaussian approximation of the maximum of two Gaussians
    # Implementation according to:
    # Sinha et al.; Advances in Computation of the Maximum of a Set of Random Variables
    a_sqr = v1 + v2 + epsilon
    a = tf.sqrt(a_sqr)
    alpha = (m1 - m2) / a
    aux_erf = tf.math.erf(alpha * (0.5 ** 0.5))
    cdf_alpha_pos = 0.5 * (1.0 + aux_erf)
    cdf_alpha_neg = 0.5 * (1.0 - aux_erf)
    pdf_norm = 1.0 / (2.0 * math.pi) ** 0.5
    pdf_alpha = pdf_norm * tf.exp(-0.5 * tf.square(alpha))
    a_times_pdf_alpha = a * pdf_alpha
     
    m_max = m1 * cdf_alpha_pos + m2 * cdf_alpha_neg + a_times_pdf_alpha
    v_max = (v1 + tf.square(m1)) * cdf_alpha_pos \
          + (v2 + tf.square(m2)) * cdf_alpha_neg \
          + (m1 + m2) * a_times_pdf_alpha \
          - tf.square(m_max) + epsilon

    return m_max, v_max


class DistPool2D(tf.keras.layers.Layer):

    def __init__(self,
                 mode,
                 pool_size,
                 strides=None,
                 padding='VALID',
                 data_format='NHWC',
                 epsilon=1e-6):
        super(DistPool2D, self).__init__()

        mode = mode.upper()
        padding = padding.upper()
        if strides is None:
            strides = pool_size

        assert mode in ['MAX', 'MAX_SIGN']
        assert padding in ['VALID', 'SAME']
        assert data_format == 'NHWC'

        self.mode= mode
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.epsilon = epsilon


    def call(self, x, training, enable_ema_updates=True):
        if training:
            if isinstance(x, tuple):
                assert len(x) == 2
                return self.call_train_distribution(x[0], x[1], enable_ema_updates)
            else:
                return self.call_train_deterministic(x, enable_ema_updates)
        else:
            return self.call_predict(x)

    
    def call_train_deterministic(self, x_in, enable_ema_updates):
        assert self.mode in ['MAX', 'MAX_SIGN']
        return tf.nn.max_pool(x_in, self.pool_size, self.strides, padding=self.padding, data_format=self.data_format)


    def call_train_distribution(self, x_in_mean, x_in_var, enable_ema_updates):
        assert self.data_format == 'NHWC'
        N, H, W, C = x_in_mean.shape[0], x_in_mean.shape[1], x_in_mean.shape[2], x_in_mean.shape[3]
        pool_H, pool_W = self.pool_size
        assert self.pool_size == self.strides
        if self.padding == 'SAME':
            assert H % pool_H == 0 and W % pool_W == 0
        elif self.padding == 'VALID':
            if H % pool_H != 0 and W % pool_W != 0:
                x_in_mean = x_in_mean[:, :-(H % pool_H), :-(W % pool_W), :] # assuming NHWC format
                x_in_var = x_in_var[:, :-(H % pool_H), :-(W % pool_W), :] # assuming NHWC format
            elif H % pool_H != 0:
                x_in_mean = x_in_mean[:, :-(H % pool_H), :, :] # assuming NHWC format
                x_in_var = x_in_var[:, :-(H % pool_H), :, :] # assuming NHWC format
            elif W % pool_W != 0:
                x_in_mean = x_in_mean[:, :, :-(W % pool_W), :] # assuming NHWC format
                x_in_var = x_in_var[:, :, :-(W % pool_W), :] # assuming NHWC format
            H, W = H - H % pool_H, W - W % pool_W
        else:
            raise NotImplementedError('DistPool2D: Padding \'{}\' not implemented'.format(self.padding))
        
        if self.mode == 'MAX':
            m, v = tf.transpose(x_in_mean, [0, 3, 1, 2]), tf.transpose(x_in_var, [0, 3, 1, 2]) # NCHW
            m, v = tf.reshape(m, [-1, pool_W]), tf.reshape(v, [-1, pool_W])
            m, v = maxOfGaussians(m[:, 0], v[:, 0], m[:, 1], v[:, 1], epsilon=self.epsilon)
            m, v = tf.reshape(m, [N, C, H, W // pool_W]), tf.reshape(v, [N, C, H, W // pool_W])
            m, v = tf.transpose(m, [0, 1, 3, 2]), tf.transpose(v, [0, 1, 3, 2]) # NCWH
            m, v = tf.reshape(m, [-1, pool_H]), tf.reshape(v, [-1, pool_H])
            m, v = maxOfGaussians(m[:, 0], v[:, 0], m[:, 1], v[:, 1], epsilon=self.epsilon)
            m, v = tf.reshape(m, [N, C, W // pool_W, H // pool_H]), tf.reshape(v, [N, C, W // pool_W, H // pool_H])
            x_out_mean, x_out_var = tf.transpose(m, [0, 3, 2, 1]), tf.transpose(v, [0, 3, 2, 1]) # NHWC
        elif self.mode == 'MAX_SIGN':
            # Note: We also tried first transposing with [0, 3, 2, 1] instead of [0, 3, 1, 2] and we found [0, 3, 1, 2]
            # to be marginally faster.
            m = (1.0 - x_in_mean) * 0.5 # Convert m to p(w = -1)
            m = tf.transpose(m, [0, 3, 1, 2]) # NCHW
            m = tf.reshape(m, [-1, pool_W])
            m = m[:, 0] * m[:, 1]
            m = tf.reshape(m, [N, C, H, W // pool_W])
            m = tf.transpose(m, [0, 1, 3, 2]) # NCWH
            m = tf.reshape(m, [-1, pool_H])
            m = m[:, 0] * m[:, 1]
            m = 1.0 - m
            m = tf.reshape(m, [N, C, W // pool_W, H // pool_H])
            m = tf.transpose(m, [0, 3, 2, 1]) # NHWC
            x_out_mean, x_out_var = m, 1.0 - tf.square(m) + max(1e-6, self.epsilon)
        else:
            raise NotImplementedError('DistPool2D: Pooling mode \'{}\' not implemented'.format(self.mode))
            
        return x_out_mean, x_out_var


    def call_predict(self, x_in):
        assert self.mode in ['MAX', 'MAX_SIGN']
        return tf.nn.max_pool(x_in, self.pool_size, self.strides, padding=self.padding, data_format=self.data_format)
