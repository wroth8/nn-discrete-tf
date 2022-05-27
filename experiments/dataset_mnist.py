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


def downloadMnist(filename):
    '''
    Downloads the MNIST data set and stores it to the given file in npz format
    '''
    from urllib.request import urlretrieve
    from gzip import open as gzopen
    from shutil import copyfileobj
    from struct import unpack
    from os import remove

    url_xTrain = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_tTrain = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    url_xTest = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_tTest = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'

    tmp_xTrain = 'tmp_mnist_xTrain.gz'
    tmp_tTrain = 'tmp_mnist_tTrain.gz'
    tmp_xTest = 'tmp_mnist_xTest.gz'
    tmp_tTest = 'tmp_mnist_tTest.gz'

    print('Downloading MNIST train images...')
    urlretrieve(url_xTrain, tmp_xTrain)
    print('Download MNIST train labels...')
    urlretrieve(url_tTrain, tmp_tTrain)
    print('Download MNIST test images...')
    urlretrieve(url_xTest, tmp_xTest)
    print('Download MNIST test labels...')
    urlretrieve(url_tTest, tmp_tTest)
    print('Downloading finished')

    print('Uncompressing gz files...')
    with gzopen(tmp_xTrain, 'rb') as f_in, open(tmp_xTrain[:-3], 'wb') as f_out:
        copyfileobj(f_in, f_out)
    with gzopen(tmp_tTrain, 'rb') as f_in, open(tmp_tTrain[:-3], 'wb') as f_out:
        copyfileobj(f_in, f_out)
    with gzopen(tmp_xTest, 'rb') as f_in, open(tmp_xTest[:-3], 'wb') as f_out:
        copyfileobj(f_in, f_out)
    with gzopen(tmp_tTest, 'rb') as f_in, open(tmp_tTest[:-3], 'wb') as f_out:
        copyfileobj(f_in, f_out)

    print('Loading uncompressed data...')
    with open(tmp_xTrain[:-3], 'rb') as f:
        magic, nImages, nRows, nCols = unpack('>IIII', f.read(16))
        assert magic == 2051
        assert nImages == 60000
        assert nRows == 28
        assert nCols == 28
        x_tr_np = np.fromfile(f, dtype=np.uint8).reshape(nImages, nRows*nCols).astype(np.float32) / 255.

    with open(tmp_tTrain[:-3], 'rb') as f:
        magic, nImages = unpack('>II', f.read(8))
        assert magic == 2049
        assert nImages == 60000
        t_tr_np = np.fromfile(f, dtype=np.int8)

    with open(tmp_xTest[:-3], 'rb') as f:
        magic, nImages, nRows, nCols = unpack('>IIII', f.read(16))
        assert magic == 2051
        assert nImages == 10000
        assert nRows == 28
        assert nCols == 28
        x_te_np = np.fromfile(f, dtype=np.uint8).reshape(nImages, nRows*nCols).astype(np.float32) / 255.

    with open(tmp_tTest[:-3], 'rb') as f:
        magic, nImages = unpack('>II', f.read(8))
        assert magic == 2049
        assert nImages == 10000
        t_te_np = np.fromfile(f, dtype=np.int8)

    x_va_np = x_tr_np[50000:]
    t_va_np = t_tr_np[50000:]
    x_tr_np = x_tr_np[:50000]
    t_tr_np = t_tr_np[:50000]

    print('Removing temporary files...')
    remove(tmp_xTrain)
    remove(tmp_xTrain[:-3])
    remove(tmp_tTrain)
    remove(tmp_tTrain[:-3])
    remove(tmp_xTest)
    remove(tmp_xTest[:-3])
    remove(tmp_tTest)
    remove(tmp_tTest[:-3])

    print('Storing MNIST data to ''%s''' % (filename))
    np.savez_compressed(filename,
                        x_tr=x_tr_np, t_tr=t_tr_np,
                        x_va=x_va_np, t_va=t_va_np,
                        x_te=x_te_np, t_te=t_te_np)

    print('MNIST is now ready')


def get_dataloader(dataset_file, batch_size=100, batch_size_eval=1000, permutation_invariant=False, training_mode=True):
    if not isfile(dataset_file):
        downloadMnist(dataset_file)

    data = dict(np.load(dataset_file))

    # Note: Inputs are already normalized to [0.0, 1.0], targets are in {0,...,9}
    data['x_tr'] = data['x_tr'].reshape(-1, 1, 28, 28) * 2.0 - 1.0
    data['x_va'] = data['x_va'].reshape(-1, 1, 28, 28) * 2.0 - 1.0
    data['x_te'] = data['x_te'].reshape(-1, 1, 28, 28) * 2.0 - 1.0

    if permutation_invariant:
        data['x_tr'] = np.reshape(data['x_tr'], (-1, 784))
        data['x_va'] = np.reshape(data['x_va'], (-1, 784))
        data['x_te'] = np.reshape(data['x_te'], (-1, 784))
    else:
        data['x_tr'] = np.transpose(data['x_tr'], (0, 2, 3, 1))
        data['x_va'] = np.transpose(data['x_va'], (0, 2, 3, 1))
        data['x_te'] = np.transpose(data['x_te'], (0, 2, 3, 1))

    data['t_tr'] = data['t_tr'].astype(np.uint8).reshape(-1, 1)
    data['t_va'] = data['t_va'].astype(np.uint8).reshape(-1, 1)
    data['t_te'] = data['t_te'].astype(np.uint8).reshape(-1, 1)

    print('MNIST dataset')
    print('x_tr.shape: {}, t_tr.shape: {}'.format(data['x_tr'].shape, data['t_tr'].shape))
    print('x_va.shape: {}, t_va.shape: {}'.format(data['x_va'].shape, data['t_va'].shape))
    print('x_te.shape: {}, t_te.shape: {}'.format(data['x_te'].shape, data['t_te'].shape))

    x_train, y_train = data['x_tr'], data['t_tr']
    x_val, y_val = data['x_va'], data['t_va']
    x_test, y_test = data['x_te'], data['t_te']

    if training_mode:
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.shuffle(x_train.shape[0])
        train_ds = train_ds.batch(batch_size, drop_remainder=True)
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
# CNN model with non piecewise constant activation function
class Cnn_Mnist_Real(Model):
    def __init__(self,
                 activation='relu',
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 batchnorm_momentum=0.99,
                 dropout_rate=None,
                 conv_kernel_size=(5,5)):
        super(Cnn_Mnist_Real, self).__init__()

        print('Cnn_Mnist_Real:')
        print('activation: \'{}\''.format(activation))
        print('regularize_weights_l1: {}'.format(regularize_weights_l1))
        print('regularize_weights_l2: {}'.format(regularize_weights_l2))
        print('batchnorm_momentum: {}'.format(batchnorm_momentum))
        print('dropout_rate: {}'.format(dropout_rate))
        print('conv_kernel_size: {}'.format(conv_kernel_size))
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

        # Layer 1: 32C-P2
        self.dropout1 = create_dropout(dropout_rate[0])
        self.conv1 = DistConv2D(32, conv_kernel_size, create_weight_type(), use_bias=False)
        self.batchnorm1 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation1 = create_activation()
        self.maxpool1 = DistPool2D('max', (2,2))
        # Layer 2: 64C-P2
        self.dropout2 = create_dropout(dropout_rate[1])
        self.conv2 = DistConv2D(64, conv_kernel_size, create_weight_type(), use_bias=False)
        self.batchnorm2 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation2 = create_activation()
        self.maxpool2 = DistPool2D('max', (2,2))
        self.flatten2 = DistFlatten()
        # Layer 3: FC512
        self.dropout3 = create_dropout(dropout_rate[2])
        self.dense3 = DistDense(512, create_weight_type(), use_bias=False)
        self.batchnorm3 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation3 = create_activation()
        # Layer 4: FC10
        self.dropout4 = create_dropout(dropout_rate[3])
        self.dense4 = DistDense(10, create_weight_type(), use_bias=True)
        self.softmax4 = Softmax()


    def call(self, x, training):
        # Layer 1: 32C-P2
        if self.dropout1 is not None:
            x = self.dropout1(x, training)
        x = self.conv1(x, training)
        x = self.batchnorm1(x, training)
        x = self.activation1(x) # TODO: Check: Do we get an error here at any time? If not, include the training argument in Cifar10/Cifar100/SVHN. We have now assertions in Activation functions
        x = self.maxpool1(x, training)
        # Layer 2: 64C-P2
        if self.dropout2 is not None:
            x = self.dropout2(x, training)
        x = self.conv2(x, training)
        x = self.batchnorm2(x, training)
        x = self.activation2(x, training)
        x = self.maxpool2(x, training)
        x = self.flatten2(x)
        # Layer 3: FC512
        if self.dropout3 is not None:
            x = self.dropout3(x, training)
        x = self.dense3(x, training)
        x = self.batchnorm3(x, training)
        x = self.activation3(x, training)
        # Layer 4: FC10
        if self.dropout4 is not None:
            x = self.dropout4(x, training)
        x = self.dense4(x, training)
        x = self.softmax4(x)
        return x


#-----------------------------------------------------------------------------------------------------------------------
# CNN model with sign activation function
class Cnn_Mnist_Discrete(Model):
    def __init__(self,
                 activation='sign',
                 initial_weight_type='real',
                 real_weights_initializer=None,
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 regularize_weights_shayer=0.0,
                 batchnorm_momentum=0.99,
                 dropout_rate=None,
                 conv_kernel_size=(5,5),
                 pool_mode='max',
                 n_samples_max_pool_reparameterization=0,
                 enable_local_reparameterization=True,
                 enable_local_reparameterization_output=True,
                 enable_gumbel_straight_through=False,
                 enable_safe_conv_variance=False,
                 enable_sampled_weights=False,
                 enable_output_activation_normalization=True):
        super(Cnn_Mnist_Discrete, self).__init__()

        print('Vgg32x32_Discrete:')
        print('activation: \'{}\''.format(activation))
        print('initial_weight_type: \'{}\''.format(initial_weight_type))
        print('real_weights_initializer: \'{}\''.format(real_weights_initializer))
        print('batchnorm_momentum: {}'.format(batchnorm_momentum))
        print('dropout_rate: {}'.format(dropout_rate))
        print('conv_kernel_size: {}'.format(conv_kernel_size))
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

        # Layer 1: 32C-P2
        self.dropout1 = create_dropout(dropout_rate[0])
        self.conv1 = DistConv2D(32, conv_kernel_size, create_weight_type(), use_bias=False, enable_safe_variance=enable_safe_conv_variance)
        self.maxpool1 = DistPool2D(pool_mode, (2,2), n_reparameterization_samples=n_samples_max_pool_reparameterization)
        self.batchnorm1 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation1 = create_activation()
        self.reparam1 = create_reparameterization()
        # Layer 2: 64C-P2
        self.dropout2 = create_dropout(dropout_rate[1])
        self.conv2 = DistConv2D(64, conv_kernel_size, create_weight_type(), use_bias=False, enable_safe_variance=enable_safe_conv_variance)
        self.maxpool2 = DistPool2D(pool_mode, (2,2), n_reparameterization_samples=n_samples_max_pool_reparameterization)
        self.batchnorm2 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation2 = create_activation()
        self.reparam2 = create_reparameterization()
        self.flatten2 = DistFlatten()
        # Layer 3: FC512
        self.dropout3 = create_dropout(dropout_rate[2])
        self.dense3 = DistDense(512, create_weight_type(), use_bias=False)
        self.batchnorm3 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation3 = create_activation()
        self.reparam3 = create_reparameterization()
        # Layer 4: FC10
        self.dropout4 = create_dropout(dropout_rate[3])
        self.dense4 = DistDense(10, create_weight_type(), use_bias=True, enable_activation_normalization=enable_output_activation_normalization)
        self.reparam4 = create_reparameterization_final()
        self.softmax4 = Softmax()


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

        # Layer 1: 32C-P2
        if self.dropout1 is not None:
            x = self.dropout1(x, training)
        x = self.conv1(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam1 is not None:
            x = self.reparam1(x, training)
        x = self.maxpool1(x, training)
        x = self.batchnorm1(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation1(x) # TODO: check if something goes wrong here, same as with CNN-Real
        if self.discrete_activation and self.reparam1 is not None:
            x = self.reparam1(x, training)

        # Layer 2: 64C-P2
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
        x = self.flatten2(x)

        # Layer 3: FC512
        if self.dropout3 is not None:
            x = self.dropout3(x, training)
        x = self.dense3(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam3 is not None:
            x = self.reparam3(x, training)
        x = self.batchnorm3(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation3(x, training)
        if self.discrete_activation and self.reparam3 is not None:
            x = self.reparam3(x, training)

        # Layer 4: FC10
        if self.dropout4 is not None:
            x = self.dropout4(x, training)
        x = self.dense4(x, training, use_sampled_weights=use_sampled_weights)

        if self.reparam4 is not None:
            x = self.reparam4(x, training)
            x_var = None
        else:
            if isinstance(x, tuple):
                assert len(x) == 2
                x_mean, x_var = x[0], x[1]
                x = x_mean
            else:
                x_var = None

        if enable_output_softmax:
            x = self.softmax4(x)

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
        w3_discrete.initialize_weights(self.dense3.dist_weights.shape, self.dense3.dist_weights)
        self.dense3.dist_weights = w3_discrete

        w4_discrete = create_weight_type()
        w4_discrete.initialize_weights(self.dense4.dist_weights.shape, self.dense4.dist_weights)
        self.dense4.dist_weights = w4_discrete


    def resample_weights(self):
        self.conv1.resample_weights()
        self.conv2.resample_weights()
        self.dense3.resample_weights()
        self.dense4.resample_weights()


#-----------------------------------------------------------------------------------------------------------------------
# Dense (fully connected) model with non piecewise constant activation function
class Dense_Mnist_Real(Model):
    def __init__(self,
                 activation='relu',
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 batchnorm_momentum=0.99,
                 dropout_rate=None):
        super(Dense_Mnist_Real, self).__init__()

        print('Dense_Mnist_Real:')
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

        # Layer 1: FC1200
        self.dropout1 = create_dropout(dropout_rate[0])
        self.dense1 = DistDense(1200, create_weight_type(), use_bias=False)
        self.batchnorm1 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation1 = create_activation()
        # Layer 2: FC1200
        self.dropout2 = create_dropout(dropout_rate[1])
        self.dense2 = DistDense(1200, create_weight_type(), use_bias=False)
        self.batchnorm2 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation2 = create_activation()
        # Layer 3: FC10
        self.dropout3 = create_dropout(dropout_rate[2])
        self.dense3 = DistDense(10, create_weight_type(), use_bias=True)
        self.softmax3 = Softmax()


    def call(self, x, training):
        # Layer 1: FC1200
        if self.dropout1 is not None:
            x = self.dropout1(x, training)
        x = self.dense1(x, training)
        x = self.batchnorm1(x, training)
        x = self.activation1(x, training)
        # Layer 2: FC1200
        if self.dropout2 is not None:
            x = self.dropout2(x, training)
        x = self.dense2(x, training)
        x = self.batchnorm2(x, training)
        x = self.activation2(x, training)
        # Layer 3: FC10
        if self.dropout3 is not None:
            x = self.dropout3(x, training)
        x = self.dense3(x, training)
        x = self.softmax3(x)
        return x


#-----------------------------------------------------------------------------------------------------------------------
# Dense (fully connected) model with sign activation function
class Dense_Mnist_Discrete(Model):
    def __init__(self,
                 activation='sign',
                 initial_weight_type='real',
                 real_weights_initializer=None,
                 regularize_weights_l1=0.0,
                 regularize_weights_l2=0.0,
                 regularize_weights_shayer=0.0,
                 batchnorm_momentum=0.99,
                 dropout_rate=None,
                 enable_local_reparameterization=True,
                 enable_local_reparameterization_output=True,
                 enable_gumbel_straight_through=False,
                 enable_sampled_weights=False,
                 enable_output_activation_normalization=True):
        super(Dense_Mnist_Discrete, self).__init__()

        print('Dense_Mnist_Discrete:')
        print('activation: \'{}\''.format(activation))
        print('initial_weight_type: \'{}\''.format(initial_weight_type))
        print('real_weights_initializer: \'{}\''.format(real_weights_initializer))
        print('batchnorm_momentum: {}'.format(batchnorm_momentum))
        print('dropout_rate: {}'.format(dropout_rate))
        print('enable_local_reparameterization: {}'.format(enable_local_reparameterization))
        print('enable_local_reparameterization_output: {}'.format(enable_local_reparameterization_output))
        print('enable_gumbel_straight_through: {}'.format(enable_gumbel_straight_through))
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

        # Layer 1: FC1200
        self.dropout1 = create_dropout(dropout_rate[0])
        self.dense1 = DistDense(1200, create_weight_type(), use_bias=False)
        self.batchnorm1 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation1 = create_activation()
        self.reparam1 = create_reparameterization()
        # Layer 2: FC1200
        self.dropout2 = create_dropout(dropout_rate[1])
        self.dense2 = DistDense(1200, create_weight_type(), use_bias=False)
        self.batchnorm2 = DistBatchNormalization(momentum=batchnorm_momentum)
        self.activation2 = create_activation()
        self.reparam2 = create_reparameterization()
        # Layer 3: FC10
        self.dropout3 = create_dropout(dropout_rate[2])
        self.dense3 = DistDense(10, create_weight_type(), use_bias=True, enable_activation_normalization=enable_output_activation_normalization)
        self.reparam3 = create_reparameterization_final()
        self.softmax3 = Softmax()


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

        # Layer 1: FC1200
        if self.dropout1 is not None:
            x = self.dropout1(x, training)
        x = self.dense1(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam1 is not None:
            x = self.reparam1(x, training)
        x = self.batchnorm1(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation1(x, training)
        if self.discrete_activation and self.reparam1 is not None:
            x = self.reparam1(x, training)

        # Layer 2: FC1200
        if self.dropout2 is not None:
            x = self.dropout2(x, training)
        x = self.dense2(x, training, use_sampled_weights=use_sampled_weights)
        if not self.discrete_activation and self.reparam2 is not None:
            x = self.reparam2(x, training)
        x = self.batchnorm2(x, training_batchnorm, compute_batchnorm_updates)
        x = self.activation2(x, training)
        if self.discrete_activation and self.reparam2 is not None:
            x = self.reparam2(x, training)

        # Layer 3: FC10
        if self.dropout3 is not None:
            x = self.dropout3(x, training)
        x = self.dense3(x, training, use_sampled_weights=use_sampled_weights)

        if self.reparam3 is not None:
            x = self.reparam3(x, training)
            x_var = None
        else:
            if isinstance(x, tuple):
                assert len(x) == 2
                x_mean, x_var = x[0], x[1]
                x = x_mean
            else:
                x_var = None

        if enable_output_softmax:
            x = self.softmax3(x)

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
        w1_discrete.initialize_weights(self.dense1.dist_weights.shape, self.dense1.dist_weights)
        self.dense1.dist_weights = w1_discrete

        w2_discrete = create_weight_type()
        w2_discrete.initialize_weights(self.dense2.dist_weights.shape, self.dense2.dist_weights)
        self.dense2.dist_weights = w2_discrete

        w3_discrete = create_weight_type()
        w3_discrete.initialize_weights(self.dense3.dist_weights.shape, self.dense3.dist_weights)
        self.dense3.dist_weights = w3_discrete


    def resample_weights(self):
        self.dense1.resample_weights()
        self.dense2.resample_weights()
        self.dense3.resample_weights()
