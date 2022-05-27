import numpy as np
import tensorflow as tf

from os.path import isfile
from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax

from layers.DistBatchNormalization import DistBatchNormalization
from layers.DistConv2D import DistConv2D
from layers.DistDense import DistDense
from layers.DistDropout import DistDropout
from layers.DistFlatten import DistFlatten
from layers.DistPool2D import DistPool2D
from layers.DistReLU import DistReLU
from layers.DistReparameterization import DistReparameterization
from layers.DistSign import DistSign
from layers.DistTanh import DistTanh
from layers.weights.QuaternaryWeights import QuaternaryWeights
from layers.weights.QuinaryWeights import QuinaryWeights
from layers.weights.RealWeights import RealWeights
from layers.weights.TernaryWeights import TernaryWeights
from layers.weights.TernaryWeightsShayer import TernaryWeightsShayer


# The following function is essentially taken from https://www.cs.toronto.edu/~kriz/cifar.html
def load_pickled(filename):
    from pickle import load as pickleload
    with open(filename, 'rb') as f:
        dct = pickleload(f, encoding='bytes')
    return dct


def downloadCifar100(filename):
    from urllib.request import urlretrieve
    from tarfile import open as taropen
    from os import remove, rmdir

    url_cifar100 = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    tmp_cifar100 = 'cifar-100-python.tar.gz'

    print('Downloading Cifar-100 dataset...')
    urlretrieve(url_cifar100, tmp_cifar100)

    print('Uncompressing tar.gz files...')
    tar = taropen(tmp_cifar100, 'r:gz')
    tar.extract('cifar-100-python/train')
    tar.extract('cifar-100-python/test')
    tar.close()

    data_train = load_pickled('cifar-100-python/train')
    data_test = load_pickled('cifar-100-python/test')

    x_tr = np.stack(data_train[b'data'], axis=0)[:45000]
    t_tr = np.asarray(data_train[b'fine_labels'])[:45000]
    x_va = np.stack(data_train[b'data'], axis=0)[45000:]
    t_va = np.asarray(data_train[b'fine_labels'])[45000:]
    x_te = np.stack(data_test[b'data'], axis=0)
    t_te = np.asarray(data_test[b'fine_labels'])

    remove('cifar-100-python/train')
    remove('cifar-100-python/test')
    rmdir('cifar-100-python')
    remove(tmp_cifar100)

    print('Storing Cifar-100 data to ''%s''' % (filename))
    np.savez_compressed(filename,
                        x_tr=x_tr, t_tr=t_tr,
                        x_va=x_va, t_va=t_va,
                        x_te=x_te, t_te=t_te)
    print('Cifar-100 is now ready')


def get_dataloader(dataset_file, batch_size=100, batch_size_eval=100, training_mode=True):
    if not isfile(dataset_file):
        downloadCifar100(dataset_file)

    data = dict(np.load(dataset_file))
    data['x_tr'] = ((data['x_tr'] / 255.0 * 2.0) - 1.0).astype(np.float32).reshape(-1, 3, 32, 32)
    data['x_va'] = ((data['x_va'] / 255.0 * 2.0) - 1.0).astype(np.float32).reshape(-1, 3, 32, 32)
    data['x_te'] = ((data['x_te'] / 255.0 * 2.0) - 1.0).astype(np.float32).reshape(-1, 3, 32, 32)

    data['x_tr'] = np.transpose(data['x_tr'], (0, 2, 3, 1))
    data['x_va'] = np.transpose(data['x_va'], (0, 2, 3, 1))
    data['x_te'] = np.transpose(data['x_te'], (0, 2, 3, 1))

    data['t_tr'] = data['t_tr'].astype(np.uint8).reshape(-1, 1)
    data['t_va'] = data['t_va'].astype(np.uint8).reshape(-1, 1)
    data['t_te'] = data['t_te'].astype(np.uint8).reshape(-1, 1)

    print('Cifar-100 dataset')
    print('x_tr.shape: {}, t_tr.shape: {}'.format(data['x_tr'].shape, data['t_tr'].shape))
    print('x_va.shape: {}, t_va.shape: {}'.format(data['x_va'].shape, data['t_va'].shape))
    print('x_te.shape: {}, t_te.shape: {}'.format(data['x_te'].shape, data['t_te'].shape))

    x_train, y_train = data['x_tr'], data['t_tr']
    x_val, y_val = data['x_va'], data['t_va']
    x_test, y_test = data['x_te'], data['t_te']

    if training_mode:
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
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.batch(batch_size_eval)
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.batch(batch_size_eval)
    val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size_eval)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds, test_ds


#-----------------------------------------------------------------------------------------------------------------------
# VGG model with non piecewise constant activation function
class Vgg32x32_Real(Model):
    def __init__(self,
                 activation='relu',
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 batchnorm_momentum=0.99,
                 dropout_rate=None):
        super(Vgg32x32_Real, self).__init__()

        print('Vgg32x32_Real:')
        print('activation: \'{}\''.format(activation))
        print('regularize_weights_l1: {}'.format(regularize_weights_l1))
        print('regularize_weights_l2: {}'.format(regularize_weights_l2))
        print('batchnorm_momentum: {}'.format(batchnorm_momentum))
        print('dropout_rate: {}'.format(dropout_rate))
        print('-' * 80)

        assert dropout_rate is not None

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
        self.dense8 = DistDense(100, create_weight_type(), use_bias=True)
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
# VGG model with sign activation function
class Vgg32x32_Discrete(Model):
    def __init__(self,
                 activation='sign',
                 initial_weight_type='real',
                 real_weights_initializer=None,
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 regularize_weights_shayer=0.0,
                 batchnorm_momentum=0.99,
                 dropout_rate=None,
                 pool_mode='max',
                 n_samples_max_pool_reparameterization=0,
                 enable_local_reparameterization=True,
                 enable_local_reparameterization_output=True,
                 enable_gumbel_straight_through=False,
                 enable_safe_conv_variance=False,
                 enable_sampled_weights=False,
                 enable_output_activation_normalization=True):
        super(Vgg32x32_Discrete, self).__init__()

        print('Vgg32x32_Discrete:')
        print('activation: \'{}\''.format(activation))
        print('initial_weight_type: \'{}\''.format(initial_weight_type))
        print('real_weights_initializer: \'{}\''.format(real_weights_initializer))
        print('batchnorm_momentum: {}'.format(batchnorm_momentum))
        print('dropout_rate: {}'.format(dropout_rate))
        print('pool_mode: \'{}\''.format(pool_mode))
        print('n_samples_max_pool_reparameterization: {}'.format(n_samples_max_pool_reparameterization))
        print('enable_local_reparameterization: {}'.format(enable_local_reparameterization))
        print('enable_local_reparameterization_output: {}'.format(enable_local_reparameterization_output))
        print('enable_gumbel_straight_through: {}'.format(enable_gumbel_straight_through))
        print('enable_safe_conv_variance: {}'.format(enable_safe_conv_variance))
        print('enable_sampled_weights: {}'.format(enable_sampled_weights))
        print('enable_output_activation_normalization: {}'.format(enable_output_activation_normalization))
        print('-' * 80)

        assert dropout_rate is not None

        if initial_weight_type == 'real':
            create_weight_type = lambda : RealWeights(
                    initializer=real_weights_initializer,
                    regularize_l1=regularize_weights_l1,
                    regularize_l2=regularize_weights_l2)
        elif initial_weight_type == 'ternary':
            create_weight_type = lambda : TernaryWeights(
                    regularize_shayer=regularize_weights_shayer,
                    q_logit_constraints=(-5.0, 5.0),
                    enable_sampled_weights=enable_sampled_weights)
        elif initial_weight_type == 'ternary_shayer':
            create_weight_type = lambda : TernaryWeightsShayer(
                    regularize_shayer=regularize_weights_shayer,
                    q_logit_constraints=(-5.0, 5.0),
                    enable_sampled_weights=enable_sampled_weights)
        elif initial_weight_type == 'quaternary':
            create_weight_type = lambda : QuaternaryWeights(
                    regularize_shayer=regularize_weights_shayer,
                    q_logit_constraints=(-5.0, 5.0),
                    enable_sampled_weights=enable_sampled_weights)
        elif initial_weight_type == 'quinary':
            create_weight_type = lambda : QuinaryWeights(
                    regularize_shayer=regularize_weights_shayer,
                    q_logit_constraints=(-5.0, 5.0),
                    enable_sampled_weights=enable_sampled_weights)
        else:
            raise NotImplementedError('Weight type \'{}\' not implemented'.format(initial_weight_type))

        assert activation in ['relu', 'tanh', 'sign'] # can be adapted later
        if activation == 'relu':
            create_activation = lambda : DistReLU()
            self.discrete_activation = False
        elif activation == 'tanh':
            create_activation = lambda : DistTanh()
            self.discrete_activation = False
        elif activation == 'sign':
            create_activation = lambda : DistSign()
            self.discrete_activation = True
        else:
            raise NotImplementedError('Unsupported activation \'{}\''.format(activation))

        self.enable_local_reparameterization = enable_local_reparameterization
        if enable_local_reparameterization:
            if activation == 'sign':
                create_reparameterization = lambda : DistReparameterization(
                        mode='GUMBEL_SOFTMAX_SIGN',
                        gumbel_softmax_temperature=1.0,
                        enable_straight_through_estimator=enable_gumbel_straight_through)
            elif activation in ['relu', 'tanh']:
                create_reparameterization = lambda : DistReparameterization(mode='NORMAL')
            else:
                raise NotImplementedError('Activation function \'{}\' not implemented for reparameterization'.format(activation))
        else:
            create_reparameterization = lambda : None

        self.enable_local_reparameterization_output = enable_local_reparameterization_output
        if enable_local_reparameterization_output:
            create_reparameterization_final = lambda : DistReparameterization(mode='NORMAL')
        else:
            create_reparameterization_final = lambda : None

        create_dropout = lambda dropout_rate : DistDropout(dropout_rate=dropout_rate, scale_at_training=True) if dropout_rate > 0.0 else None

        # Layer 1: 128C3
        self.dropout1 = create_dropout(dropout_rate[0])
        self.conv1 = DistConv2D(128, (3,3), create_weight_type(), use_bias=False, enable_safe_variance=enable_safe_conv_variance)
        self.batchnorm1 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation1 = create_activation()
        self.reparam1 = create_reparameterization()
        # Layer 2: 128C3-P2
        self.dropout2 = create_dropout(dropout_rate[1])
        self.conv2 = DistConv2D(128, (3,3), create_weight_type(), use_bias=False, enable_safe_variance=enable_safe_conv_variance)
        self.maxpool2 = DistPool2D(pool_mode, (2,2), n_reparameterization_samples=n_samples_max_pool_reparameterization)
        self.batchnorm2 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation2 = create_activation()
        self.reparam2 = create_reparameterization()

        # Layer 3: 256C3
        self.dropout3 = create_dropout(dropout_rate[2])
        self.conv3 = DistConv2D(256, (3,3), create_weight_type(), use_bias=False, enable_safe_variance=enable_safe_conv_variance)
        self.batchnorm3 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation3 = create_activation()
        self.reparam3 = create_reparameterization()
        # Layer 4: 256C3-P2
        self.dropout4 = create_dropout(dropout_rate[3])
        self.conv4 = DistConv2D(256, (3,3), create_weight_type(), use_bias=False, enable_safe_variance=enable_safe_conv_variance)
        self.maxpool4 = DistPool2D(pool_mode, (2,2), n_reparameterization_samples=n_samples_max_pool_reparameterization)
        self.batchnorm4 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation4 = create_activation()
        self.reparam4 = create_reparameterization()

        #  Layer 5: 512C3
        self.dropout5 = create_dropout(dropout_rate[4])
        self.conv5 = DistConv2D(512, (3,3), create_weight_type(), use_bias=False, enable_safe_variance=enable_safe_conv_variance)
        self.batchnorm5 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation5 = create_activation()
        self.reparam5 = create_reparameterization()
        # Layer 6: 512C3-P2-Flatten
        self.dropout6 = create_dropout(dropout_rate[5])
        self.conv6 = DistConv2D(512, (3,3), create_weight_type(), use_bias=False, enable_safe_variance=enable_safe_conv_variance)
        self.maxpool6 = DistPool2D(pool_mode, (2,2), n_reparameterization_samples=n_samples_max_pool_reparameterization)
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
        self.dense8 = DistDense(100, create_weight_type(), use_bias=True, enable_activation_normalization=enable_output_activation_normalization)
        self.reparam8 = create_reparameterization_final()
        self.softmax8 = Softmax()


    def call(self, x, training, compute_prediction_updates=False, return_output_variance=False, enable_output_softmax=True, use_sampled_weights=False):
        '''
        compute_prediction_updates: If set to True, batchnorm does not compute updates during training and instead
          computes updates during prediction. Keep in mind that this flag must not be set if for inference. It is
          intended to compute updates using the prediction path which often results in substantially different
          activation statistics. By default the updates are computed during training. The following if-cascade shows
          best how to use these flags.
        '''
        assert training or not return_output_variance # return_output_variance can only be set if training == True

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
        x = self.conv1(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam1 is not None:
            x = self.reparam1(x, training)
        x = self.batchnorm1(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation1(x)
        if self.discrete_activation and self.reparam1 is not None:
            x = self.reparam1(x, training)

        # Layer 2: 128C3-P2
        if self.dropout2 is not None:
            x = self.dropout2(x, training)
        x = self.conv2(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam2 is not None:
            x = self.reparam2(x, training)
        x = self.maxpool2(x, training)
        x = self.batchnorm2(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation2(x, training)
        if self.discrete_activation and self.reparam2 is not None:
            x = self.reparam2(x, training)

        # Layer 3: 256C3
        if self.dropout3 is not None:
            x = self.dropout3(x, training)
        x = self.conv3(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam3 is not None:
            x = self.reparam3(x, training)
        x = self.batchnorm3(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation3(x, training)
        if self.discrete_activation and self.reparam3 is not None:
            x = self.reparam3(x, training)

        # Layer 4: 256C3-P2
        if self.dropout4 is not None:
            x = self.dropout4(x, training)
        x = self.conv4(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam4 is not None:
            x = self.reparam4(x, training)
        x = self.maxpool4(x, training)
        x = self.batchnorm4(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation4(x, training)
        if self.discrete_activation and self.reparam4 is not None:
            x = self.reparam4(x, training)

        #  Layer 5: 512C3
        if self.dropout5 is not None:
            x = self.dropout5(x, training)
        x = self.conv5(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam5 is not None:
            x = self.reparam5(x, training)
        x = self.batchnorm5(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation5(x, training)
        if self.discrete_activation and self.reparam5 is not None:
            x = self.reparam5(x, training)

        # Layer 6: 512C3-P2-Flatten
        if self.dropout6 is not None:
            x = self.dropout6(x, training)
        x = self.conv6(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam6 is not None:
            x = self.reparam6(x, training)
        x = self.maxpool6(x, training)
        x = self.batchnorm6(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation6(x, training)
        if self.discrete_activation and self.reparam6 is not None:
            x = self.reparam6(x, training)
        x = self.flatten6(x)

        # Layer 7: FC1024
        if self.dropout7 is not None:
            x = self.dropout7(x, training)
        x = self.dense7(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam7 is not None:
            x = self.reparam7(x, training)
        x = self.batchnorm7(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation7(x, training)
        if self.discrete_activation and self.reparam7 is not None:
            x = self.reparam7(x, training)

        # Layer 8: FC10
        if self.dropout8 is not None:
            x = self.dropout8(x, training)
        x = self.dense8(x, training, use_sampled_weights=use_sampled_weights)

        if self.reparam8 is not None:
            x = self.reparam8(x, training)
            x_var = None
        else:
            if isinstance(x, tuple):
                assert len(x) == 2
                x_mean, x_var = x[0], x[1]
                x = x_mean
            else:
                x_var = None

        if enable_output_softmax:
            x = self.softmax8(x)

        if training and return_output_variance:
            return x, x_var # returns softmax(x_mean), x_var [or for enable_output_softmax==False: x_mean, x_var]
        else:
            return x


    def replace_weights(self, weight_type, regularize_shayer, init_mode, enable_sampled_weights=False):
        assert weight_type in ['ternary', 'ternary_shayer', 'quaternary', 'quinary']

        if weight_type == 'ternary':
            create_weight_type = lambda : TernaryWeights(
                    regularize_shayer=regularize_shayer,
                    q_logit_constraints=(-5.0, 5.0),
                    initializer_mode=init_mode,
                    enable_sampled_weights=enable_sampled_weights)
        elif weight_type == 'ternary_shayer':
            create_weight_type = lambda : TernaryWeightsShayer(
                    regularize_shayer=regularize_shayer,
                    q_logit_constraints=(-5.0, 5.0),
                    initializer_mode=init_mode,
                    enable_sampled_weights=enable_sampled_weights)
        elif weight_type == 'quaternary':
            create_weight_type = lambda : QuaternaryWeights(
                    regularize_shayer=regularize_shayer,
                    q_logit_constraints=(-5.0, 5.0),
                    enable_sampled_weights=enable_sampled_weights)
        elif weight_type == 'quinary':
            create_weight_type = lambda : QuinaryWeights(
                    regularize_shayer=regularize_shayer,
                    q_logit_constraints=(-5.0, 5.0),
                    enable_sampled_weights=enable_sampled_weights)
        else:
            raise NotImplementedError('WeightType \'{}\' not implemented'.format(weight_type))

        w1_discrete = create_weight_type()
        w1_discrete.initialize_weights(self.conv1.dist_weights.shape, self.conv1.dist_weights)
        self.conv1.dist_weights = w1_discrete

        w2_discrete = create_weight_type()
        w2_discrete.initialize_weights(self.conv2.dist_weights.shape, self.conv2.dist_weights)
        self.conv2.dist_weights = w2_discrete

        w3_discrete = create_weight_type()
        w3_discrete.initialize_weights(self.conv3.dist_weights.shape, self.conv3.dist_weights)
        self.conv3.dist_weights = w3_discrete

        w4_discrete = create_weight_type()
        w4_discrete.initialize_weights(self.conv4.dist_weights.shape, self.conv4.dist_weights)
        self.conv4.dist_weights = w4_discrete

        w5_discrete = create_weight_type()
        w5_discrete.initialize_weights(self.conv5.dist_weights.shape, self.conv5.dist_weights)
        self.conv5.dist_weights = w5_discrete

        w6_discrete = create_weight_type()
        w6_discrete.initialize_weights(self.conv6.dist_weights.shape, self.conv6.dist_weights)
        self.conv6.dist_weights = w6_discrete

        w7_discrete = create_weight_type()
        w7_discrete.initialize_weights(self.dense7.dist_weights.shape, self.dense7.dist_weights)
        self.dense7.dist_weights = w7_discrete

        w8_discrete = create_weight_type()
        w8_discrete.initialize_weights(self.dense8.dist_weights.shape, self.dense8.dist_weights)
        self.dense8.dist_weights = w8_discrete


    def resample_weights(self):
        self.conv1.resample_weights()
        self.conv2.resample_weights()
        self.conv3.resample_weights()
        self.conv4.resample_weights()
        self.conv5.resample_weights()
        self.conv6.resample_weights()
        self.dense7.resample_weights()
        self.dense8.resample_weights()
