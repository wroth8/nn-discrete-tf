'''
This file contains quantization functions with straight-through gradient estimators
'''

import tensorflow as tf


@tf.custom_gradient
def sign0_ste_id(x):
    '''
    Sign function with identity straight-through estimator. Note that sign(0) = 0.
    '''
    y = tf.math.sign(x)
    def grad(dy):
        return dy
    return y, grad


@tf.custom_gradient
def sign0_ste_tanh(x):
    '''
    Sign function with tanh straight-through estimator. Note that sign(0) = 0.
    '''
    y = tf.math.sign(x)
    def grad(dy):
        tanh_x = tf.math.tanh(x)
        return dy * (1.0 - tf.square(tanh_x))
    return y, grad


@tf.custom_gradient
def sign0_ste_hardtanh(x):
    '''
    Sign function with hard-tanh straight-through estimator. Note that sign(0) = 0.
    '''
    y = tf.math.sign(x)
    def grad(dy):
        return tf.where(tf.abs(x) <= 1.0, dy, 0.0)
    return y, grad


@tf.custom_gradient
def sign_ste_id(x):
    '''
    Sign function with identity straight-through estimator. Note that sign(0) = 1.
    '''
    y = tf.cast(tf.where(x >= 0.0, 1.0, -1.0), x.dtype)
    def grad(dy):
        return dy
    return y, grad


@tf.custom_gradient
def sign_ste_tanh(x):
    '''
    Sign function with tanh straight-through estimator. Note that sign(0) = 1.
    '''
    y = tf.cast(tf.where(x >= 0.0, 1.0, -1.0), x.dtype)
    def grad(dy):
        tanh_x = tf.math.tanh(x)
        return dy * (1.0 - tf.square(tanh_x))
    return y, grad


@tf.custom_gradient
def sign_ste_hardtanh(x):
    '''
    Sign function with hard-tanh straight-through estimator. Note that sign(0) = 1.
    '''
    y = tf.cast(tf.where(x >= 0.0, 1.0, -1.0), x.dtype)
    def grad(dy):
        return tf.where(tf.abs(x) <= 1.0, dy, 0.0)
    return y, grad


@tf.custom_gradient
def sign_stoch_ste_id(x):
    '''
    Stochastic sign quantizer with identity straight-through estimator
    '''
    prob_pos = (x + 1.0) * 0.5 # do not need clipping to [0,1] here
    U = tf.random.uniform(x.shape, dtype=x.dtype)
    y = tf.where(U <= prob_pos, 1.0, -1.0)
    def grad(dy):
        return dy
    return y, grad


@tf.custom_gradient
def sign_stoch_ste_tanh(x):
    '''
    Stochastic sign quantizer with tanh straight-through estimator
    '''
    prob_pos = (x + 1.0) * 0.5 # do not need clipping to [0,1] here
    U = tf.random.uniform(x.shape, dtype=x.dtype)
    y = tf.where(U <= prob_pos, 1.0, -1.0)
    def grad(dy):
        tanh_x = tf.math.tanh(x)
        return dy * (1.0 - tf.square(tanh_x))
    return y, grad


@tf.custom_gradient
def sign_stoch_ste_hardtanh(x):
    '''
    Stochastic sign quantizer with hard-tanh straight-through estimator
    '''
    prob_pos = (x + 1.0) * 0.5 # do not need clipping to [0,1] here
    U = tf.random.uniform(x.shape, dtype=x.dtype)
    y = tf.where(U <= prob_pos, 1.0, -1.0)
    def grad(dy):
        return tf.where(tf.abs(x) <= 1.0, dy, 0.0)
    return y, grad


@tf.custom_gradient
def sign_dorefa(x):
    '''
    The binary quantizer used in the DoReFa-Net paper
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients, arXiv:1606.06160 (2016)
    '''
    y = tf.cast(tf.where(x >= 0.0, 1.0, -1.0), x.dtype) * tf.reduce_mean(tf.abs(x))
    def grad(dy):
        return dy
    return y, grad


@tf.custom_gradient
def linear_quantizer(x, num_bits, min_x, max_x):
    '''
    Linear quantizer according to
    Miyashita et al., Convolutional Neural Networks using Logarithmic Data Representation.
    For symmetric limits (min_x, max_x) around zero, this quantizer also has an output 0. However, the largest value
    is not attained by this method.
    Example: num_bits=2, min_x=-1, max_x=1 => {-1, -0.5, 0, 0.5}
    '''
    num_vals = 2.0 ** num_bits
    step = 2.0 ** -num_bits
    x = (x - min_x) * (num_vals / (max_x - min_x)) # transform to [0,num_vals]
    x = tf.math.round(x)
    x = tf.clip_by_value(x, 0, num_vals - 1.0) # clip values
    x = x * (step * (max_x - min_x)) + min_x
    def grad(dy):
        return dy, None, None, None
    return x, grad


@tf.custom_gradient
def linear_quantizer_dorefa(x, num_bits, min_x, max_x):
    '''
    Linear quantizer according to
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients, arXiv:1606.06160 (2016)
    This quantizer quantized the weights evently between min_x and max_x. This function is called quantize_k in the
    paper.
    Example: num_bits=2, min_x=-1, max_x=1 => {-1, -1/3, 1/3, 1}
    '''
    x = (x - min_x) * ((2.0 ** num_bits - 1.0) / (max_x - min_x)); # transform to [0,2**num_bits-1]
    x = tf.math.round(x) * (1.0 / (2.0 ** num_bits - 1.0));
    x = tf.clip_by_value(x, 0, 1);
    x = x * (max_x - min_x) + min_x;
    def grad(dy):
        return dy, None, None, None
    return x, grad


def tanh_quantizer_dorefa(x, num_bits):
    '''
    Tanh quantizer: According to Equation (9) from
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients, arXiv:1606.06160 (2016)
    '''
    x = tf.math.tanh(x) # transform to [-1, 1]
    x = x * (0.5 / tf.reduce_max(tf.abs(x))) + 0.5 # transform to [0, 1]
    x = linear_quantizer_dorefa(x, num_bits, -1.0, 1.0) # STE is implemented here
    x = x * 2.0 - 1.0
    return x
