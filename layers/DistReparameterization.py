import tensorflow as tf


class DistReparameterization(tf.keras.layers.Layer):

    def __init__(self, mode='NORMAL', gumbel_softmax_temperature=1.0, epsilon=1e-6):
        super(DistReparameterization, self).__init__()
        mode = mode.upper()
        assert mode in ['NORMAL', 'GUMBEL_SOFTMAX_SIGN', 'GUMBEL_SOFTMAX_BINARY']
        self.mode = mode
        self.gumbel_softmax_temperature = 1.0
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
        Warning('Layer \'DistReparameterization\' applied to deterministic input has no effect')
        return x_in


    def call_train_distribution(self, x_in_mean, x_in_var):
        if self.mode == 'NORMAL':
            noise = tf.random.normal(x_in_mean.shape)
            x_out = x_in_mean + noise * tf.sqrt(x_in_var + self.epsilon)
        elif self.mode == 'GUMBEL_SOFTMAX_SIGN':
            probs = tf.stack([-x_in_mean, x_in_mean], axis=0) + 1.0 # Factor 0.5 omitted since its an irrelevant additive constant for the logits
            logits = tf.math.log(probs + max(1e-6, self.epsilon))
            noise_uniform = tf.random.uniform(logits.shape)
            noise_gumbel = -tf.math.log(-tf.math.log(noise_uniform + self.epsilon) + self.epsilon) # Gumbel(0, 1) noise
            logits_sample = logits + noise_gumbel
            logits_delta = (logits_sample[1,...] - logits_sample[0,...]) * (0.5 / self.gumbel_softmax_temperature)
            x_out = tf.math.tanh(logits_delta)
        elif self.mode == 'GUMBEL_SOFTMAX_BINARY':
            raise NotImplementedError('DistReparameterization: Reparameterization mode \'{}\' not tested'.format(self.mode))
            probs = T.stack([1.0 - x_in_mean, x_in_mean], axis=0)
            logits = T.log(probs + max(1e-6, self.epsilon))
            noise_uniform = tf.random.uniform(logits.shape)
            noise_gumbel = -tf.math.log(-tf.math.log(noise_uniform + self.epsilon) + self.epsilon) # Gumbel(0, 1) noise
            logits_sample = logits + noise_gumbel
            logits_delta = (logits_sample[1,...] - logits_sample[0,...]) * (1.0 / self.gumbel_softmax_temperature)
            x_out = tf.math.sigmoid(logits_delta)
            # OLD: Theano implementation
            # aux = T.stack([1.-x_in_mean, x_in_mean], axis=0)
            # logits = T.log(aux + max(1e-6, self._epsilon_reparameterization))
            # U = self._srng.uniform(logits.shape, dtype=theano.config.floatX)
            # gumbel_epsilon = -T.log(-T.log(U + self._epsilon_reparameterization) + self._epsilon_reparameterization)
            # logits_sample = (logits + gumbel_epsilon) * (1. / self._gumbel_softmax_temperature)
            # x_out = T.nnet.sigmoid(logits_sample[1,...] - logits_sample[0,...])
            # if self._enable_straight_through_gumbel_estimator:
            # x_out = threshold_half_st(x_out)
        else:
            raise NotImplementedError('DistReparameterization: Reparameterization mode \'{}\' not implemented'.format(self.mode))
        return x_out


    def call_predict(self, x_in):
        return x_in
