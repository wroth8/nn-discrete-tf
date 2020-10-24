import tensorflow as tf
import numpy as np

from time import time

from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax

from layers.DistDropout import DistDropout
from layers.DistConv2D import DistConv2D
from layers.DistDense import DistDense
from layers.DistReparameterization import DistReparameterization
from layers.DistBatchNormalization import DistBatchNormalization
from layers.DistPool2D import DistPool2D
from layers.DistSign import DistSign
from layers.DistFlatten import DistFlatten

from layers.weights.RealWeights import RealWeights
from layers.weights.TernaryWeights import TernaryWeights

#-----------------------------------------------------------------------------------------------------------------------
# VGG model with sign activation function
class VggSign32x32(Model):
    def __init__(self,
                 initial_weight_type='real',
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 regularize_weights_shayer=0.0,
                 batchnorm_momentum=0.99,
                 dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0],
                 use_gumbel_softmax_reparameterization=True):
        super(VggSign32x32, self).__init__()

        if initial_weight_type == 'real':
            create_weight_type = lambda : RealWeights(regularize_l1=regularize_weights_l1,
                                                      regularize_l2=regularize_weights_l2)
        elif initial_weight_type == 'ternary':
            create_weight_type = lambda : TernaryWeights(regularize_shayer=regularize_weights_shayer)
        else:
            raise NotImplementedError('Weighty type \'{}\' not implemented'.format(weight_type))
        
        create_activation = lambda : DistSign()

        create_reparameterization = lambda : DistReparameterization(
                mode='GUMBEL_SOFTMAX_SIGN',
                gumbel_softmax_temperature=1.0) if use_gumbel_softmax_reparameterization else None
        create_reparameterization_final = lambda : DistReparameterization(mode='NORMAL')

        create_dropout = lambda dropout_rate : DistDropout(dropout_rate=dropout_rate, scale_at_training=True) if dropout_rate > 0.0 else None

        # Layer 1: 128C3
        self.dropout1 = create_dropout(dropout_rate[0])
        self.conv1 = DistConv2D(128, (3,3), create_weight_type(), use_bias=False)
        self.batchnorm1 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation1 = create_activation()
        self.reparam1 = create_reparameterization()
        # Layer 2: 128C3-P2
        self.dropout2 = create_dropout(dropout_rate[1])
        self.conv2 = DistConv2D(128, (3,3), create_weight_type(), use_bias=False)
        self.maxpool2 = DistPool2D('max', (2,2))
        self.batchnorm2 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation2 = create_activation()
        self.reparam2 = create_reparameterization()
        
        # Layer 3: 256C3
        self.dropout3 = create_dropout(dropout_rate[2])
        self.conv3 = DistConv2D(256, (3,3), create_weight_type(), use_bias=False)
        self.batchnorm3 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation3 = create_activation()
        self.reparam3 = create_reparameterization()
        # Layer 4: 256C3-P2
        self.dropout4 = create_dropout(dropout_rate[3])
        self.conv4 = DistConv2D(256, (3,3), create_weight_type(), use_bias=False)
        self.maxpool4 = DistPool2D('max', (2,2))
        self.batchnorm4 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation4 = create_activation()
        self.reparam4 = create_reparameterization()

        #  Layer 5: 512C3
        self.dropout5 = create_dropout(dropout_rate[4])
        self.conv5 = DistConv2D(512, (3,3), create_weight_type(), use_bias=False)
        self.batchnorm5 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation5 = create_activation()
        self.reparam5 = create_reparameterization()
        # Layer 6: 512C3-P2-Flatten
        self.dropout6 = create_dropout(dropout_rate[5])
        self.conv6 = DistConv2D(512, (3,3), create_weight_type(), use_bias=False)
        self.maxpool6 = DistPool2D('max', (2,2))
        self.batchnorm6 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation6 = create_activation()
        self.reparam6 = create_reparameterization()
        self.flatten6 = DistFlatten()
        # Layer 7: FC1024
        self.dropout7 = create_dropout(dropout_rate[6])
        self.dense7 = DistDense(1024, create_weight_type(), use_bias=False)
        self.batchnorm7 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation7 = create_activation()
        self.reparam7 = create_reparameterization()
        # Layer 8: FC10
        self.dropout8 = create_dropout(dropout_rate[7])
        self.dense8 = DistDense(10, create_weight_type(), use_bias=True, enable_activation_normalization=True)
        self.reparam8 = create_reparameterization_final()
        self.softmax8 = Softmax()


    def call(self, x, training, compute_prediction_updates=False):
        '''
        compute_prediction_updates: If set to True, batchnorm does not compute updates during training and instead
          computes updates during prediction. Keep in mind that this flag must not be set if for inference. It is
          intended to compute updates using the prediction path which often results in substantially different
          activation statistics. By default the updates are computed during training. The following if-cascade shows
          best how to use these flags.
        '''
        
        if training == True and compute_prediction_updates == True:
            # Compute training path without batchnorm updates
            training_batchnorm = True
            compute_batchnorm_updates = False
        elif training == True and compute_prediction_updates == False:
            # Compute training path with batchnorm updates
            training_batchnorm = True
            compute_batchnorm_updates = True
        elif training == False and compute_prediction_updates == True:
            # Compute prediction path with batchnorm updates.
            training_batchnorm = True
            compute_batchnorm_updates = True
        elif training == False and compute_prediction_updates == False:
            # Compute prediction path without batchnorm updates
            training_batchnorm = False
            compute_batchnorm_updates = False
        else:
            assert False

        # Layer 1: 128C3
        if self.dropout1 is not None:
            x = self.dropout1(x, training)
        x = self.conv1(x, training)
        x = self.batchnorm1(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation1(x)
        if self.reparam1 is not None:
            x = self.reparam1(x, training)
        # Layer 2: 128C3-P2
        if self.dropout2 is not None:
            x = self.dropout2(x, training)
        x = self.conv2(x, training)
        x = self.maxpool2(x, training)
        x = self.batchnorm2(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation2(x, training)
        if self.reparam2 is not None:
            x = self.reparam2(x, training)
        # Layer 3: 256C3
        if self.dropout3 is not None:
            x = self.dropout3(x, training)
        x = self.conv3(x, training)
        x = self.batchnorm3(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation3(x, training)
        if self.reparam3 is not None:
            x = self.reparam3(x, training)
        # Layer 4: 256C3-P2
        if self.dropout4 is not None:
            x = self.dropout4(x, training)
        x = self.conv4(x, training)
        x = self.maxpool4(x, training)
        x = self.batchnorm4(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation4(x, training)
        if self.reparam4 is not None:
            x = self.reparam4(x, training)
        #  Layer 5: 512C3
        if self.dropout5 is not None:
            x = self.dropout5(x, training)
        x = self.conv5(x, training)
        x = self.batchnorm5(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation5(x, training)
        if self.reparam5 is not None:
            x = self.reparam5(x, training)
        # Layer 6: 512C3-P2-Flatten
        if self.dropout6 is not None:
            x = self.dropout6(x, training)
        x = self.conv6(x, training)
        x = self.maxpool6(x, training)
        x = self.batchnorm6(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation6(x, training)
        if self.reparam6 is not None:
            x = self.reparam6(x, training)
        x = self.flatten6(x)
        # Layer 7: FC1024
        if self.dropout7 is not None:
            x = self.dropout7(x, training)
        x = self.dense7(x, training)
        x = self.batchnorm7(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation7(x, training)
        if self.reparam7 is not None:
            x = self.reparam7(x, training)
        # Layer 8: FC10
        if self.dropout8 is not None:
            x = self.dropout8(x, training)
        x = self.dense8(x, training)
        x = self.reparam8(x, training)
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
# Note: The initial weight type must be `real` because we load real-valued weights from the file. These weights will
# later be replaced by ternary weights.
model = VggSign32x32(initial_weight_type='real',
                     batchnorm_momentum=0.9, # We used 0.9 in the Theano implementation
                     regularize_weights_l2=1e-3,
                     dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0])

model(np.ones((2,32,32,3), dtype=np.float32), True, compute_prediction_updates=True) # Build the model
compute_prediction_updates = True # Determines whether EMA updates of batchnorm should be computed online [==False] or after every epoch

model_file_real = 'saved_models/model_cifar10_vgg32x32_real'
model_file_ternary = 'saved_models/model_cifar10_vgg32x32_ternary'

try:
    model.load_weights(model_file_real)
except:
    print('Could not read initial parameters from \'{}\''.format(model_file_real))

# The following lines initialize the ternary distributions with the pre-trained real-valued weights
create_weight_type = lambda : TernaryWeights(regularize_shayer=1e-10)
w1_ternary = create_weight_type()
w1_ternary.initialize_weights(model.conv1.dist_weights.shape, model.conv1.dist_weights)
model.conv1.dist_weights = w1_ternary

w2_ternary = create_weight_type()
w2_ternary.initialize_weights(model.conv2.dist_weights.shape, model.conv2.dist_weights)
model.conv2.dist_weights = w2_ternary

w3_ternary = create_weight_type()
w3_ternary.initialize_weights(model.conv3.dist_weights.shape, model.conv3.dist_weights)
model.conv3.dist_weights = w3_ternary

w4_ternary = create_weight_type()
w4_ternary.initialize_weights(model.conv4.dist_weights.shape, model.conv4.dist_weights)
model.conv4.dist_weights = w4_ternary

w5_ternary = create_weight_type()
w5_ternary.initialize_weights(model.conv5.dist_weights.shape, model.conv5.dist_weights)
model.conv5.dist_weights = w5_ternary

w6_ternary = create_weight_type()
w6_ternary.initialize_weights(model.conv6.dist_weights.shape, model.conv6.dist_weights)
model.conv6.dist_weights = w6_ternary

w7_ternary = create_weight_type()
w7_ternary.initialize_weights(model.dense7.dist_weights.shape, model.dense7.dist_weights)
model.dense7.dist_weights = w7_ternary

w8_ternary = create_weight_type()
w8_ternary.initialize_weights(model.dense8.dist_weights.shape, model.dense8.dist_weights)
model.dense8.dist_weights = w8_ternary

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

# Initialize the optimizer. We use two optimizers to implement two different learning rates. The learning rates of logit
# parameters are greater than the other parameters.
learning_rate_variable = tf.Variable(1e-3, tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_variable)
optimizer_logits = tf.keras.optimizers.Adam(learning_rate=learning_rate_variable*10.0)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, True, compute_prediction_updates)
        loss = loss_object(labels, predictions)
        if model.losses:
            loss += tf.add_n(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)

    updates_params = []
    updates_logit_params = []
    for gradient, trainable_variable in zip(gradients, model.trainable_variables):
        if 'Logits' not in trainable_variable.name:
            updates_params.append((gradient, trainable_variable))
        else:
            updates_logit_params.append((gradient, trainable_variable))
    print('#(non-logit parameters): {:3d}'.format(len(updates_params)))
    print('#(logit parameters):     {:3d}'.format(len(updates_logit_params)))
    optimizer.apply_gradients(updates_params)
    optimizer_logits.apply_gradients(updates_logit_params)

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def train_prediction_updates(images):
    model(images, False, compute_prediction_updates=True)


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

tensorboard_logdir = 'logs/cifar10_ternary'
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

    if compute_prediction_updates:
        # Compute batchnorm updates after epoch
        for images, _ in train_ds:
            train_prediction_updates(images)
       
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

if model_file_ternary is not None:
    model.save_weights(model_file_ternary)
