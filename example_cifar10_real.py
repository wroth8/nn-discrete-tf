'''
Example script for running a training on Cifar-10 using real weights and
activations.
'''

import numpy as np
import tensorflow as tf

from time import time

from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax

from layers.DistBatchNormalization import DistBatchNormalization
from layers.DistConv2D import DistConv2D
from layers.DistDense import DistDense
from layers.DistDropout import DistDropout
from layers.DistFlatten import DistFlatten
from layers.DistPool2D import DistPool2D
from layers.DistReLU import DistReLU
from layers.DistTanh import DistTanh
from layers.weights.RealWeights import RealWeights

#-----------------------------------------------------------------------------------------------------------------------
# VGG model with non piecewise constant activation function
class Vgg32x32(Model):
    def __init__(self,
                 activation='relu',
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 batchnorm_momentum=0.99,
                 dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0]):
        super(Vgg32x32, self).__init__()

        create_weight_type = lambda : RealWeights(regularize_l1=regularize_weights_l1,
                                                  regularize_l2=regularize_weights_l2)
        
        if activation == 'relu':
            create_activation = lambda : DistReLU()
        elif activation == 'tanh':
            create_activation = lambda : DistTanh()
        else:
            raise NotImplementedError('Activation function \'{}\' not implemented'.format(activation))
        
        create_dropout = lambda dropout_rate : DistDropout(dropout_rate=dropout_rate, scale_at_training=True) if dropout_rate > 0.0 else None

        # Layer 1: 128C3
        # self.dropout1 = create_dropout(dropout_rate[0]) if dropout_rate[0] > 0.0 else None
        self.dropout1 = create_dropout(dropout_rate[0])
        self.conv1 = DistConv2D(128, (3,3), create_weight_type(), use_bias=False)
        self.batchnorm1 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation1 = create_activation()
        # Layer 2: 128C3-P2
        self.dropout2 = create_dropout(dropout_rate[1])
        self.conv2 = DistConv2D(128, (3,3), create_weight_type(), use_bias=False)
        self.batchnorm2 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation2 = create_activation()
        self.maxpool2 = DistPool2D('max', (2,2))
        # Layer 3: 256C3
        self.dropout3 = create_dropout(dropout_rate[2])
        self.conv3 = DistConv2D(256, (3,3), create_weight_type(), use_bias=False)
        self.batchnorm3 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation3 = create_activation()
        # Layer 4: 256C3-P2
        self.dropout4 = create_dropout(dropout_rate[3])
        self.conv4 = DistConv2D(256, (3,3), create_weight_type(), use_bias=False)
        self.batchnorm4 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation4 = create_activation()
        self.maxpool4 = DistPool2D('max', (2,2))
        #  Layer 5: 512C3
        self.dropout5 = create_dropout(dropout_rate[4])
        self.conv5 = DistConv2D(512, (3,3), create_weight_type(), use_bias=False)
        self.batchnorm5 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation5 = create_activation()
        # Layer 6: 512C3-P2-Flatten
        self.dropout6 = create_dropout(dropout_rate[5])
        self.conv6 = DistConv2D(512, (3,3), create_weight_type(), use_bias=False)
        self.batchnorm6 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation6 = create_activation()
        self.maxpool6 = DistPool2D('max', (2,2))
        self.flatten6 = DistFlatten()
        # Layer 7: FC1024
        self.dropout7 = create_dropout(dropout_rate[6])
        self.dense7 = DistDense(1024, create_weight_type(), use_bias=False)
        self.batchnorm7 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation7 = create_activation()
        # Layer 8: FC10
        self.dropout8 = create_dropout(dropout_rate[7])
        self.dense8 = DistDense(10, create_weight_type(), use_bias=True)
        self.softmax8 = Softmax()


    def call(self, x, training):
        # Layer 1: 128C3
        if self.dropout1 is not None:
            x = self.dropout1(x, training)
        x = self.conv1(x, training)
        x = self.batchnorm1(x, training)
        x = self.activation1(x)
        # Layer 2: 128C3-P2
        if self.dropout2 is not None:
            x = self.dropout2(x, training)
        x = self.conv2(x, training)
        x = self.batchnorm2(x, training)
        x = self.activation2(x, training)
        x = self.maxpool2(x, training)
        # Layer 3: 256C3
        if self.dropout3 is not None:
            x = self.dropout3(x, training)
        x = self.conv3(x, training)
        x = self.batchnorm3(x, training)
        x = self.activation3(x, training)
        # Layer 4: 256C3-P2
        if self.dropout4 is not None:
            x = self.dropout4(x, training)
        x = self.conv4(x, training)
        x = self.batchnorm4(x, training)
        x = self.activation4(x, training)
        x = self.maxpool4(x, training)
        #  Layer 5: 512C3
        if self.dropout5 is not None:
            x = self.dropout5(x, training)
        x = self.conv5(x, training)
        x = self.batchnorm5(x, training)
        x = self.activation5(x, training)
        # Layer 6: 512C3-P2-Flatten
        if self.dropout6 is not None:
            x = self.dropout6(x, training)
        x = self.conv6(x, training)
        x = self.batchnorm6(x, training)
        x = self.activation6(x, training)
        x = self.maxpool6(x, training)
        x = self.flatten6(x)
        # Layer 7: FC1024
        if self.dropout7 is not None:
            x = self.dropout7(x, training)
        x = self.dense7(x, training)
        x = self.batchnorm7(x, training)
        x = self.activation7(x, training)
        # Layer 8: FC10
        if self.dropout8 is not None:
            x = self.dropout8(x, training)
        x = self.dense8(x, training)
        x = self.softmax8(x)
        return x


#-----------------------------------------------------------------------------------------------------------------------
# Data loader
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = 2.0 * x_train - 1.0, 2.0 * x_test - 1.0
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_val = x_train[45000:]
x_train = x_train[:45000]
y_val = y_train[45000:]
y_train = y_train[:45000]

def preprocess_image_train(image):
    image = tf.image.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image

class generator(object):
    def __init__(self, data):
        self.data_tuple = data
    def generate(self):
        yield self.data_tuple

gen_train = generator((x_train, y_train))
train_ds = tf.data.Dataset.from_generator(
            gen_train.generate,
            output_types=(tf.float32, tf.uint8),
            output_shapes=((45000, 32, 32, 3), (45000, 1)))
train_ds = train_ds.unbatch()
train_ds = train_ds.shuffle(45000)
train_ds = train_ds.batch(1000) # Batch here so that resize_with_crop_or_pad works more efficient
train_ds = train_ds.map(lambda img, lbl: (tf.image.resize_with_crop_or_pad(img, 32 + 8, 32 + 8), lbl))
train_ds = train_ds.unbatch()
train_ds = train_ds.map(lambda img, lbl: (preprocess_image_train(img), lbl))
train_ds = train_ds.batch(100)
train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = val_ds.batch(100)
val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.batch(100)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

#-----------------------------------------------------------------------------------------------------------------------
# Create the model
model = Vgg32x32(activation='relu',
                 batchnorm_momentum=0.99,
                 regularize_weights_l2=1e-3,
                 dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0])

model(np.ones((2,32,32,3), dtype=np.float32), False) # Build the model

model_file_real = 'saved_models/model_cifar10_vgg32x32_real' # Resulting model will be stored in this file
try:
    model.load_weights(model_file_real)
except:
    print('Could not read initial parameters from \'{}\''.format(model_file_real))


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

# Initialize the optimizer
learning_rate_variable = tf.Variable(1e-4, tf.float32)
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
EPOCHS = 300

tensorboard_logdir = 'logs/cifar10_real'
if tensorboard_logdir is not None:
    logwriter = tf.summary.create_file_writer(tensorboard_logdir)

print('Start optimization. #epochs: {}. Initial learning rate: {}'.format(EPOCHS, learning_rate_variable))

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
    if epoch > 0 and epoch % 100 == 0:
        learning_rate_variable.assign(learning_rate_variable * 0.1)
        print('Decreasing learning rate. New learning rate: {}'.format(learning_rate_variable))

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
    if tensorboard_logdir is not None:
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

if model_file_real is not None:
    model.save_weights(model_file_real)
