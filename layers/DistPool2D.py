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


def logArgmaxShekhovtsov(m, v, epsilon=1e-6):
    # The arguments m and v are assumed to be (N, D) where D is the number of Gaussians
    # This function computes for each of the N rows an approximation of the probability that
    # a particular Gaussian is the maximum. The resuts are returned as normalized
    # log-probabilities.
    # This method computes the quadratic-time approximation from
    # Shekhovtsov and Flach; Feed-forward Propagation in Probabilistic Neural Networks with Categorical and Max Layers; ICLR 2019
    sigma_S = math.pi / (3.0 ** 0.5)
    log_q_list = []
    for idx in range(m.shape[1]):
        log_q_list.append(-tf.reduce_logsumexp((m - m[:, idx, None]) * tf.math.rsqrt(v + v[:, idx, None] + epsilon) * sigma_S, axis=1)) # rsqrt, check with epsilon
    log_q = tf.stack(log_q_list, axis=1)
    log_q_normalized = log_q - tf.reduce_logsumexp(log_q, axis=1, keepdims=True) # normalize logits
    return log_q_normalized


def logArgmaxShekhovtsovFast(m, v, epsilon=1e-6):
    # The arguments m and v are assumed to be (N, D) where D is the number of Gaussians
    # This function computes for each of the N rows an approximation of the probability that
    # a particular Gaussian is the maximum. The resuts are returned as normalized
    # log-probabilities.
    # This method computes the linear-time approximation from
    # Shekhovtsov and Flach; Feed-forward Propagation in Probabilistic Neural Networks with Categorical and Max Layers; ICLR 2019
    sigma_S_inv = 3 ** 0.5 / math.pi # inverse standard deviation of the standard logistic distribution
    v_a = tf.reduce_min(v, axis=1, keepdims=True) + epsilon
    s = (sigma_S_inv * 2.0 ** 0.5) * tf.sqrt(v_a)
    s_y = tf.sqrt(v + v_a) * sigma_S_inv

    m_div_s = m / s
    # first compute sum over all k (including y)
    A = tf.reduce_logsumexp(m_div_s, axis=1, keepdims=True)  # also sums over y
    # then subtract the entry for k=y
    A_sub_y = tf.math.log(1.0 - tf.exp(m_div_s - A) + epsilon) + A # this line uses log-sum-exp trick with A being the maximum (since mu_div_s <= A)

    S_all = (-m + s * A_sub_y) / s_y
    log_q = -tf.math.softplus(S_all)
    log_q_normalized = log_q - tf.reduce_logsumexp(log_q, axis=-1, keepdims=True) # normalize logits
    return log_q_normalized


def logistic_icdf(x, epsilon=1e-6):
    # Computes the inverse cdf of the logistic distribution (or, equivalently, the inverse of the sigmoid function)
    return tf.math.log(x + epsilon) - tf.math.log1p(-x + epsilon)


def binary_entropy(q, epsilon=1e-6):
    # Computes the elementwise entropy of a Bernoulli distribution with parameter (=mean) q
    return -(q * tf.math.log(q + epsilon) + (1.0 - q) * tf.math.log1p(-q + epsilon)) + epsilon


def maxOfGaussianShekhovtsov(m, v, enable_fast_mode, epsilon=1e-6):
    # The arguments m and v are assumed to be (N, D) where D is the number of Gaussians
    # This function computes an approximation of the mean and the variance of the D Gaussians in each row.
    # If enable_fast_mode is set, the linear-time approximation from Shekhovtsov and Flach (2019) is used, otherwise
    # the quadratic-time approximation in their paper is used.
    if not enable_fast_mode:
        log_q = logArgmaxShekhovtsov(m, v, epsilon=epsilon)
    else:
        log_q = logArgmaxShekhovtsovFast(m, v, epsilon=epsilon)
    q = tf.exp(log_q) # this is already normalized

    sigma_S_inv = 3 ** 0.5 / math.pi
    aux = tf.sqrt(v + epsilon) * sigma_S_inv * binary_entropy(q, epsilon=epsilon)
    m_hat = tf.where(q >= epsilon, m + aux / tf.maximum(q, epsilon), m) # Note: tf.maximum is important to get a "good" gradient. See DistReLU for a similar issue.
    m_max = tf.reduce_sum(q * m_hat, axis=1)

    const_a = -1.33751
    const_b = 0.886763
    v_max = tf.reduce_sum(v * tf.math.sigmoid(const_a + const_b * logistic_icdf(q, epsilon=epsilon)) + q * (m_hat - m_max[:, None]) ** 2.0, axis=1)
    return m_max, v_max + epsilon


class DistPool2D(tf.keras.layers.Layer):

    def __init__(self,
                 mode,
                 pool_size,
                 strides=None,
                 padding='VALID',
                 data_format='NHWC',
                 n_reparameterization_samples=None,
                 epsilon=1e-6):
        super(DistPool2D, self).__init__()

        mode = mode.upper()
        padding = padding.upper()
        if strides is None:
            strides = pool_size

        assert mode in ['MAX', 'MAX_SIGN', 'MAX_MEAN', 'MAX_SAMPLE', 'MAX_SAMPLE_REPARAM', 'MAX_SHEKHOVTSOV', 'MAX_SHEKHOVTSOV_FAST']
        assert padding in ['VALID', 'SAME']
        assert data_format == 'NHWC'

        self.mode= mode
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.n_reparameterization_samples = n_reparameterization_samples
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
        assert self.mode in ['MAX', 'MAX_SIGN', 'MAX_MEAN', 'MAX_SAMPLE', 'MAX_SAMPLE_REPARAM', 'MAX_SHEKHOVTSOV', 'MAX_SHEKHOVTSOV_FAST']
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
            assert pool_W == 2 and pool_H == 2 # current implementation only supports 2x2 pooling
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
            assert pool_W == 2 and pool_H == 2 # current implementation only supports 2x2 pooling
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
        elif self.mode == 'MAX_MEAN':
            # Compute maximum mean of pooling region and pass corresponding mean and variance
            m, v = tf.transpose(x_in_mean, [0, 3, 1, 2]), tf.transpose(x_in_var, [0, 3, 1, 2]) # convert to NCHW
            m, v = tf.reshape(m, [N, C, H, W // pool_W, pool_W]), tf.reshape(v, [N, C, H, W // pool_W, pool_W])
            m, v = tf.transpose(m, [0, 1, 3, 4, 2]), tf.transpose(v, [0, 1, 3, 4, 2])
            assert m.shape == (N, C, W // pool_W, pool_W, H) and m.shape == v.shape

            m, v = tf.reshape(m, [N, C, W // pool_W, pool_W, H // pool_H, pool_H]), tf.reshape(v, [N, C, W // pool_W, pool_W, H // pool_H, pool_H])
            m, v = tf.transpose(m, [0, 4, 2, 1, 3, 5]), tf.transpose(v, [0, 4, 2, 1, 3, 5])
            assert m.shape == (N, H // pool_H, W // pool_W, C, pool_W, pool_H) and m.shape == v.shape

            m, v = tf.reshape(m, [-1, pool_W * pool_H]), tf.reshape(v, [-1, pool_W * pool_H])
            max_idx = tf.argmax(m, axis=1) # select argmax along last dimension
            m, v = tf.gather(m, max_idx, axis=1, batch_dims=1), tf.gather(v, max_idx, axis=1, batch_dims=1)
            x_out_mean, x_out_var = tf.reshape(m, [N, H // pool_H, W // pool_W, C]), tf.reshape(v, [N, H // pool_H, W // pool_W, C]) # convert back to NHWC
        elif self.mode == 'MAX_SAMPLE':
            # Sample according to mean and variance, compute maximum sample of pooling region, and pass mean and variance of maximum sample
            # This method is used in "J.W.T. Peters and M. Welling: Probabilistic binary neural networks, 2018"
            m, v = tf.transpose(x_in_mean, [0, 3, 1, 2]), tf.transpose(x_in_var, [0, 3, 1, 2]) # convert to NCHW
            m, v = tf.reshape(m, [N, C, H, W // pool_W, pool_W]), tf.reshape(v, [N, C, H, W // pool_W, pool_W])
            m, v = tf.transpose(m, [0, 1, 3, 4, 2]), tf.transpose(v, [0, 1, 3, 4, 2])
            assert m.shape == (N, C, W // pool_W, pool_W, H) and m.shape == v.shape

            m, v = tf.reshape(m, [N, C, W // pool_W, pool_W, H // pool_H, pool_H]), tf.reshape(v, [N, C, W // pool_W, pool_W, H // pool_H, pool_H])
            m, v = tf.transpose(m, [0, 4, 2, 1, 3, 5]), tf.transpose(v, [0, 4, 2, 1, 3, 5])
            assert m.shape == (N, H // pool_H, W // pool_W, C, pool_W, pool_H) and m.shape == v.shape

            m, v = tf.reshape(m, [-1, pool_W * pool_H]), tf.reshape(v, [-1, pool_W * pool_H])
            sample = tf.random.normal(m.shape) * tf.sqrt(v) + m
            sample = tf.stop_gradient(sample) # treat sample as a constant and do not backpropagate through it
            max_idx = tf.argmax(sample, axis=1) # select argmax of the sample along last dimension
            m, v = tf.gather(m, max_idx, axis=1, batch_dims=1), tf.gather(v, max_idx, axis=1, batch_dims=1)
            x_out_mean, x_out_var = tf.reshape(m, [N, H // pool_H, W // pool_W, C]), tf.reshape(v, [N, H // pool_H, W // pool_W, C]) # convert back to NHWC
        elif self.mode == 'MAX_SAMPLE_REPARAM':
            # Computes a Monte-Carlo estimate of the mean and variance of the maximum of the Gaussians of each pooling
            # region. To be compatible with backpropagation, the reparameterization trick is used to sample from the
            # Gaussians.
            assert self.n_reparameterization_samples is not None
            assert self.n_reparameterization_samples >= 2

            m, v = tf.reshape(x_in_mean, [N, H // pool_H, pool_H, W // pool_W, pool_W, C]), tf.reshape(x_in_var, [N, H // pool_H, pool_H, W // pool_W, pool_W, C]) # input must be NHWC
            m, v = tf.transpose(m, [0, 1, 3, 5, 2, 4]), tf.transpose(v, [0, 1, 3, 5, 2, 4])
            m, v = tf.reshape(m, [-1, pool_H * pool_W]), tf.reshape(v, [-1, pool_H * pool_W])

            if self.n_reparameterization_samples == 2:
                m, v = tf.reshape(m, [-1, pool_W * pool_H]), tf.reshape(v, [-1, pool_W * pool_H])
                sqrt_v = tf.sqrt(v + self.epsilon)
                sample1 = tf.random.normal(m.shape) * sqrt_v + m
                sample1_max = tf.reduce_max(sample1, axis=1)
                sample2 = tf.random.normal(m.shape) * sqrt_v + m
                sample2_max = tf.reduce_max(sample2, axis=1)
                m = 0.5 * (sample1_max + sample2_max)
                v = 0.5 * tf.square(sample1_max - sample2_max) + self.epsilon
            else:
                m, v = tf.reshape(m, [-1, pool_W * pool_H]), tf.reshape(v, [-1, pool_W * pool_H])
                sqrt_v = tf.sqrt(v + self.epsilon)
                sample = tf.random.normal((self.n_reparameterization_samples,) + m.shape) * sqrt_v + m
                sample_max = tf.reduce_max(sample, axis=2)
                m = tf.reduce_mean(sample_max, axis=0)
                v = tf.reduce_sum(tf.square(sample_max - m), axis=0) * (1.0 / (self.n_reparameterization_samples - 1.0)) + self.epsilon
            x_out_mean, x_out_var = tf.reshape(m, [N, H // pool_H, W // pool_W, C]), tf.reshape(v, [N, H // pool_H, W // pool_W, C]) # convert back to NHWC
        elif self.mode in ['MAX_SHEKHOVTSOV', 'MAX_SHEKHOVTSOV_FAST']:
            # Max-Pooling according to
            # Shekhovtsov and Flach; Feed-forward Propagation in Probabilistic Neural Networks with Categorical and Max Layers; ICLR 2019
            # The paper proposes to methods:
            # - MAX_SHEKHOVTSOV: The quadratic-time method in their paper
            # - MAX_SHEKHOVTSOV_FAST: The linear-time method in their paper (they conduct experiments using this method in the paper)
            m, v = tf.reshape(x_in_mean, [N, H // pool_H, pool_H, W // pool_W, pool_W, C]), tf.reshape(x_in_var, [N, H // pool_H, pool_H, W // pool_W, pool_W, C]) # input must be NHWC
            m, v = tf.transpose(m, [0, 1, 3, 5, 2, 4]), tf.transpose(v, [0, 1, 3, 5, 2, 4])
            m, v = tf.reshape(m, [-1, pool_H * pool_W]), tf.reshape(v, [-1, pool_H * pool_W])

            if self.mode == 'MAX_SHEKHOVTSOV':
                m, v = maxOfGaussianShekhovtsov(m, v, False, epsilon=self.epsilon)
            elif self.mode == 'MAX_SHEKHOVTSOV_FAST':
                m, v = maxOfGaussianShekhovtsov(m, v, True, epsilon=self.epsilon)
            else:
                raise NotImplementedError('Unknown mode \'{}\'. We should not be here.'.format(self.mode))
            x_out_mean, x_out_var = tf.reshape(m, [N, H // pool_H, W // pool_W, C]), tf.reshape(v, [N, H // pool_H, W // pool_W, C]) # convert back to NHWC
        else:
            raise NotImplementedError('DistPool2D: Pooling mode \'{}\' not implemented'.format(self.mode))

        return x_out_mean, x_out_var


    def call_predict(self, x_in):
        assert self.mode in ['MAX', 'MAX_SIGN', 'MAX_MEAN', 'MAX_SAMPLE', 'MAX_SAMPLE_REPARAM', 'MAX_SHEKHOVTSOV', 'MAX_SHEKHOVTSOV_FAST']
        return tf.nn.max_pool(x_in, self.pool_size, self.strides, padding=self.padding, data_format=self.data_format)
