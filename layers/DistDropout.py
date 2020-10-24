import tensorflow as tf


class DistDropout(tf.keras.layers.Layer):

    def __init__(self,
                 dropout_rate=0.1,
                 scale_at_training=True):
        '''
        Dropout Layer:
          @see Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov;
               Dropout: A Simple Way to Prevent Neural Networks from Overfitting
               JMLR 15(Jun):1929−1958, 2014
        
        Arguments:
        dropout_rate: The probability of an input being set to zero.
        scale_at_training: Determines whether the outputs should be scaled at training or at inference.
        '''
        super(DistDropout, self).__init__()
        assert 0.0 <= dropout_rate and dropout_rate < 1.0
        self.dropout_rate = dropout_rate
        self.dropout_keep_prob = 1.0 - dropout_rate
        self.dropout_scale = 1.0 / (self.dropout_keep_prob)
        self.dropout_scale_sq = self.dropout_scale ** 2.0
        self.scale_at_training = scale_at_training


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
        dropout_mask = tf.random.uniform(x_in.shape) >= self.dropout_rate
        if self.scale_at_training == True:
            dropout_mask = tf.where(dropout_mask, self.dropout_scale, 0.0)
        else:
            dropout_mask = tf.cast(dropout_mask, x_in.dtype)
        return x_in * dropout_mask


    def call_train_distribution(self, x_in_mean, x_in_var):
        dropout_mask = tf.cast(tf.random.uniform(x_in_mean.shape) >= self.dropout_rate, x_in_mean.dtype)
        if self.scale_at_training == True:
            x_out_mean = x_in_mean * self.dropout_scale * dropout_mask
            x_out_var = x_in_var * self.dropout_scale_sq * dropout_mask
        else:
            x_out_mean = x_in_mean * dropout_mask
            x_out_var = x_in_var * dropout_mask
        return x_out_mean, x_out_var


    def call_predict(self, x_in):
        if self.scale_at_training == True:
            x_out = x_in
        else:
            x_out = x_in * self.dropout_keep_prob
        return x_out
