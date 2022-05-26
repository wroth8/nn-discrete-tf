'''
Example script for running a training on MNIST using binary weights and
activations using the straight-through gradient estimator.
'''

import numpy as np
import tensorflow as tf

from time import time

from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax

from layers.DistBatchNormalization import DistBatchNormalization
from layers.DistDense import DistDense
from layers.DistDropout import DistDropout
from layers.DistReLU import DistReLU
from layers.DistSign import DistSign
from layers.ste import sign0_ste_id
from layers.weights.QuantizedWeightsStraightThrough import QuantizedWeightsStraightThrough
from layers.weights.RealWeights import RealWeights

#-----------------------------------------------------------------------------------------------------------------------
# Fully connected model with support for quantized weights and activations using the straight-through gradient estimator
class MnistFullyConnectedNN(Model):
    def __init__(self,
                 weight_type='real',
                 activation='relu',
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 batchnorm_momentum=0.99,
                 dropout_rate=[0.2,0.4,0.4]):
        super(MnistFullyConnectedNN, self).__init__()

        self.batchnorm_momentum = batchnorm_momentum
        if weight_type == 'real':
            create_weights = lambda : RealWeights(regularize_l1=regularize_weights_l1,
                                                  regularize_l2=regularize_weights_l2)
        elif weight_type == 'binary_ste':
            create_weights = lambda : QuantizedWeightsStraightThrough(sign0_ste_id,
                                                                      regularize_l1=regularize_weights_l1,
                                                                      regularize_l2=regularize_weights_l2)
        else:
            raise NotImplementedError('Weighty type \'{}\' not implemented'.format(weight_type))
        
        if activation == 'relu':
            create_activation = lambda : DistReLU()
        elif activation == 'sign':
            create_activation = lambda : DistSign(has_zero_output=False, straight_through_type='tanh')
        else:
            raise NotImplementedError('Activation \'{}\' not implemented'.format(activation))

        create_dropout = lambda dropout_rate : DistDropout(dropout_rate=dropout_rate, scale_at_training=True) if dropout_rate > 0.0 else None

        # Layer 1
        self.dropout1 = create_dropout(dropout_rate[0])
        self.dense1 = DistDense(1200, create_weights(), use_bias=False)
        self.batchnorm1 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.act1 = create_activation()
        # Layer 2
        self.dropout2 = create_dropout(dropout_rate[1])
        self.dense2 = DistDense(1200, create_weights(), use_bias=False)
        self.batchnorm2 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.act2 = create_activation()
        # Layer 3
        self.dropout3 = create_dropout(dropout_rate[2])
        self.dense3 = DistDense(10, create_weights(), use_bias=True)
        self.softmax3 = Softmax()

    
    def call(self, x, training):
        # Layer 1
        if self.dropout1 is not None:
            x = self.dropout1(x, training)
        x = self.dense1(x, training)
        x = self.batchnorm1(x, training)
        x = self.act1(x, training)
        # Layer 2
        if self.dropout2 is not None:
            x = self.dropout2(x, training)
        x = self.dense2(x, training)
        x = self.batchnorm2(x, training)
        x = self.act2(x, training)
        # Layer 3
        if self.dropout3 is not None:
            x = self.dropout3(x, training)
        x = self.dense3(x, training)
        x = self.softmax3(x)
        return x

#-----------------------------------------------------------------------------------------------------------------------
# Data loader
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

x_val = x_train[50000:]
x_train = x_train[:50000]
y_val = y_train[50000:]
y_train = y_train[:50000]

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(50000).batch(100).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(1000).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000).prefetch(tf.data.experimental.AUTOTUNE)

#-----------------------------------------------------------------------------------------------------------------------
# Create the model
enable_binary_weights = True
enable_binary_activations = True
model = MnistFullyConnectedNN(weight_type='binary_ste' if enable_binary_weights else 'real',
                              activation='sign' if enable_binary_activations else 'relu',
                              batchnorm_momentum=0.99 if (not enable_binary_weights and not enable_binary_activations) else 0.9,
                              regularize_weights_l2=1e-5,
                              dropout_rate=[0.2, 0.4, 0.4])

model(np.ones((2,784), dtype=np.float32), True) # Build the model

model_file_real_relu = 'saved_models/model_mnist_fc_real_relu'
model_file_real_sign = 'saved_models/model_mnist_fc_real_sign'
model_file_binary_relu = 'saved_models/model_mnist_fc_binary_relu'
model_file_binary_sign = 'saved_models/model_mnist_fc_binary_sign'

if enable_binary_weights and enable_binary_activations:
    result_file = model_file_binary_sign
elif enable_binary_weights and not enable_binary_activations:
    result_file = model_file_binary_relu
elif not enable_binary_weights and enable_binary_activations:
    result_file = model_file_real_sign
else:
    result_file = model_file_real_relu

# Uncomment the following lines to use a pre-trained real-valued model for initialization
# try:
#     model.load_weights(model_file_real_relu)
# except:
#     print('Could not read initial parameters from \'{}\''.format(model_file_real_relu))

print('#' * 80)
model.summary()
print('#' * 80)

#-----------------------------------------------------------------------------------------------------------------------
# Create TF functions
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

if enable_binary_weights and enable_binary_activations:
    learning_rate_variable = tf.Variable(3e-3, tf.float32) # suitable for binary weights and binary activation
    learning_rate_schedule = np.logspace(np.log10(3e-3), np.log10(3e-7), 1000)
else:
    learning_rate_variable = tf.Variable(3e-4, tf.float32) # suitable for real weights and/or relu activation
    learning_rate_schedule = np.arange(0.0, 10.0).reshape(-1, 1) # decrease every 250 iterations by factor of 10
    learning_rate_schedule = np.repeat(learning_rate_schedule, 250, axis=1).reshape(-1)
    learning_rate_schedule = 3e-4 * (10 ** -learning_rate_schedule)
    
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_variable)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, True)
        loss = loss_object(labels, predictions)
        if model.losses:
            loss += tf.add_n(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def train_prediction_updates(images):
    model(images, False)


@tf.function
def val_step(images, labels):
    predictions = model(images, False)
    t_loss = loss_object(labels, predictions)

    val_loss(t_loss)
    val_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

#-----------------------------------------------------------------------------------------------------------------------
# Optimization
EPOCHS = 1000

enable_tensorboard_logging = True
if enable_tensorboard_logging:
    log_dir = 'logs/mnist_fc_binary_ste'
    logwriter = tf.summary.create_file_writer(log_dir)

# Compute initial validation and test errors, respectively.
t0_eval_val = time()
for val_images, val_labels in val_ds:
    val_step(val_images, val_labels)
t_elapsed_eval_val = time() - t0_eval_val

t0_eval_test = time()
for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)
t_elapsed_eval_test = time() - t0_eval_test

# Convert tensors to python scalars
to_pyscalar = lambda tensor : tensor.numpy().item()
pyval_val_loss = to_pyscalar(val_loss.result())
pyval_val_error = to_pyscalar(1.0 - val_accuracy.result())
pyval_test_loss = to_pyscalar(test_loss.result())
pyval_test_error = to_pyscalar(1.0 - test_accuracy.result())

template = 'Epoch {:3d}/{:3d}, Loss: {:12e}, CE[TR]: {:8.5f}, Loss[VA]: {:e}, CE[VA]: {:8.5f}, Loss[TE]: {:e}, CE[TE]: {:8.5f}'
print(template.format(0,
                      EPOCHS,
                      float('nan'),
                      float('nan'),
                      pyval_val_loss,
                      pyval_val_error * 100.0,
                      pyval_test_loss,
                      pyval_test_error * 100.0))
val_loss.reset_states()
val_accuracy.reset_states()
test_loss.reset_states()
test_accuracy.reset_states()

for epoch in range(EPOCHS):
    learning_rate_variable.assign(learning_rate_schedule[epoch])
    if epoch % 100 == 0:
        print('Current learning rate: {}'.format(learning_rate_variable))

    t0_train = time()
    for images, labels in train_ds:
        train_step(images, labels)
    t_elapsed_train = time() - t0_train

    t0_eval_val = time()
    for val_images, val_labels in val_ds:
        val_step(val_images, val_labels)
    t_elapsed_eval_val = time() - t0_eval_val

    t0_eval_test = time()
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    t_elapsed_eval_test = time() - t0_eval_test

    # Convert tensors to python scalars
    pyval_train_loss = to_pyscalar(train_loss.result())
    pyval_train_error = to_pyscalar(1.0 - train_accuracy.result())
    pyval_val_loss = to_pyscalar(val_loss.result())
    pyval_val_error = to_pyscalar(1.0 - val_accuracy.result())
    pyval_test_loss = to_pyscalar(test_loss.result())
    pyval_test_error = to_pyscalar(1.0 - test_accuracy.result())
    pyval_learning_rate = to_pyscalar(learning_rate_variable)

    template = 'Epoch {:3d}/{:3d}, Loss: {:e}, CE[TR]: {:8.5f}, Loss[VA]: {:e}, CE[VA]: {:8.5f}, Loss[TE]: {:e}, CE[TE]: {:8.5f} [t_elapsed: {:6.2f} seconds]'
    print(template.format(epoch+1,
                          EPOCHS,
                          pyval_train_loss,
                          pyval_train_error * 100.0,
                          pyval_val_loss,
                          pyval_val_error * 100.0,
                          pyval_test_loss,
                          pyval_test_error * 100.0,
                          t_elapsed_train + t_elapsed_eval_val + t_elapsed_eval_test))

    # Write interesting values to tensorboard
    if enable_tensorboard_logging:
        with logwriter.as_default():
            tf.summary.scalar("train_loss", pyval_train_loss, step=epoch+1)
            tf.summary.scalar("train_error", pyval_train_error * 100.0, step=epoch+1)
            tf.summary.scalar("val_loss", pyval_val_loss, step=epoch+1)
            tf.summary.scalar("val_error", pyval_val_error * 100.0, step=epoch+1)
            tf.summary.scalar("test_loss", pyval_test_loss, step=epoch+1)
            tf.summary.scalar("test_error", pyval_test_error * 100.0, step=epoch+1)
            tf.summary.scalar("t_elapsed_train", t_elapsed_train, step=epoch+1)
            tf.summary.scalar("t_elapsed_eval_val", t_elapsed_eval_val, step=epoch+1)
            tf.summary.scalar("t_elapsed_eval_test", t_elapsed_eval_test, step=epoch+1)
            tf.summary.scalar("learning_rate", pyval_learning_rate, step=epoch+1)

    # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

model.save_weights(result_file)
