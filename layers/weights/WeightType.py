import tensorflow as tf

# class WeightType(tf.Module):
class WeightType(tf.keras.layers.Layer):
    def __init__(self):
        super(WeightType, self).__init__()

    def initialize_weights(self):
        raise NotImplementedError('\'initialize_weights()\' must be implemented by subclass')

    def apply_losses(self):
        raise NotImplementedError('\'apply_losses()\' must be implemented by subclass')

    def is_deterministic(self):
        raise NotImplementedError('\'is_deterministic()\' must be implemented by subclass')

    def expectation(self):
        raise NotImplementedError('\'expectation()\' must be implemented by subclass')

    def variance(self):
        raise NotImplementedError('\'variance()\' must be implemented by subclass')

    def expectation_and_variance(self):
        raise NotImplementedError('\'expectation_and_variance()\' must be implemented by subclass')
    
    def most_probable(self):
        raise NotImplementedError('\'most_probable()\' must be implemented by subclass')

    def resample_weights(self):
        raise NotImplementedError('\'resample_weights()\' must be implemented by subclass')

    def sampled(self):
        raise NotImplementedError('\'sampled()\' must be implemented by subclass')