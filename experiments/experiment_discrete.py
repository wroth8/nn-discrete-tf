import numpy as np
import tensorflow as tf

from optparse import OptionParser
from os.path import isfile
from time import time


def run(dataset,
        dataset_file,
        init_weight_type,
        init_mode_real,
        init_mode_discrete,
        weight_type,
        activation,
        enable_local_reparameterization,
        enable_local_reparameterization_output,
        enable_gumbel_straight_through,
        n_epochs,
        batch_size,
        lr_init,
        lr_init_logits,
        lr_schedule_reduce_factor,
        lr_schedule_reduce_every,
        reg_logits_l2,
        reg_kl, # TODO: argument not implemented
        batchnorm_momentum,
        batchnorm_reestimation,
        dropout_rate,
        pool_mode,
        n_samples_max_pool_reparameterization,
        enable_safe_conv_variance,
        model_file_init,
        model_file_best,
        model_file_last,
        tensorboard_logdir):
    #-----------------------------------------------------------------------------------------------------------------------
    # Note: For init_weight_type `real` we initialize with real-valued weights (typically from a stored model).
    # These weights will later be replaced by the given weight_type.
    if init_mode_real == 'file':
        real_weights_initializer = None
    elif init_mode_real == 'normal':
        real_weights_initializer = lambda shape, fan_in, fan_out : tf.random.normal(shape, 0.0, 1.0)
    elif init_mode_real == 'uniform':
        real_weights_initializer = lambda shape, fan_in, fan_out : tf.random.uniform(shape, -1.0, 1.0)
    elif init_mode_real == 'equidist':
        real_weights_initializer = lambda shape, fan_in, fan_out : tf.reshape(tf.random.shuffle(tf.linspace(-1.0, 1.0, np.prod(shape).item())), shape)
    elif init_mode_real == 'equidist_ternary':
        real_weights_initializer = lambda shape, fan_in, fan_out : tf.reshape(tf.random.shuffle(tf.linspace(-1.5, 1.5, np.prod(shape).item())), shape)
    else:
        raise NotImplementedError('init_mode_real \'{}\' not supported'.format(init_mode_real))

    if dataset == 'cifar10':
        from dataset_cifar10 import Vgg32x32_Discrete, get_dataloader
        model = Vgg32x32_Discrete(
                activation=activation,
                initial_weight_type=init_weight_type,
                real_weights_initializer=real_weights_initializer,
                regularize_weights_shayer=reg_logits_l2,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0] if dropout_rate is None else dropout_rate,
                pool_mode=pool_mode,
                n_samples_max_pool_reparameterization=n_samples_max_pool_reparameterization,
                enable_local_reparameterization=enable_local_reparameterization,
                enable_local_reparameterization_output=enable_local_reparameterization_output,
                enable_gumbel_straight_through=enable_gumbel_straight_through,
                enable_safe_conv_variance=enable_safe_conv_variance)
        dummy_input = np.ones((2,32,32,3), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size=batch_size)
    elif dataset == 'cifar100':
        from dataset_cifar100 import Vgg32x32_Discrete, get_dataloader
        model = Vgg32x32_Discrete(
                activation=activation,
                initial_weight_type=init_weight_type,
                real_weights_initializer=real_weights_initializer,
                regularize_weights_shayer=reg_logits_l2,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0] if dropout_rate is None else dropout_rate,
                pool_mode=pool_mode,
                n_samples_max_pool_reparameterization=n_samples_max_pool_reparameterization,
                enable_local_reparameterization=enable_local_reparameterization,
                enable_local_reparameterization_output=enable_local_reparameterization_output,
                enable_gumbel_straight_through=enable_gumbel_straight_through,
                enable_safe_conv_variance=enable_safe_conv_variance)
        dummy_input = np.ones((2,32,32,3), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size=batch_size)
    elif dataset == 'svhn':
        from dataset_svhn import Vgg32x32_Discrete, get_dataloader
        model = Vgg32x32_Discrete(
                activation=activation,
                initial_weight_type=init_weight_type,
                real_weights_initializer=real_weights_initializer,
                regularize_weights_shayer=reg_logits_l2,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0] if dropout_rate is None else dropout_rate,
                pool_mode=pool_mode,
                n_samples_max_pool_reparameterization=n_samples_max_pool_reparameterization,
                enable_local_reparameterization=enable_local_reparameterization,
                enable_local_reparameterization_output=enable_local_reparameterization_output,
                enable_gumbel_straight_through=enable_gumbel_straight_through,
                enable_safe_conv_variance=enable_safe_conv_variance)
        dummy_input = np.ones((2,32,32,3), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size=batch_size)
    elif dataset == 'mnist':
        from dataset_mnist import Cnn_Mnist_Discrete, get_dataloader
        model = Cnn_Mnist_Discrete(
                activation=activation,
                initial_weight_type=init_weight_type,
                real_weights_initializer=real_weights_initializer,
                regularize_weights_shayer=reg_logits_l2,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.0, 0.2, 0.3, 0.0] if dropout_rate is None else dropout_rate,
                conv_kernel_size=(5,5),
                pool_mode=pool_mode,
                n_samples_max_pool_reparameterization=n_samples_max_pool_reparameterization,
                enable_local_reparameterization=enable_local_reparameterization,
                enable_local_reparameterization_output=enable_local_reparameterization_output,
                enable_gumbel_straight_through=enable_gumbel_straight_through,
                enable_safe_conv_variance=enable_safe_conv_variance)
        dummy_input = np.ones((2,28,28,1), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size=batch_size, permutation_invariant=False)
    elif dataset == 'mnist-pi':
        from dataset_mnist import Dense_Mnist_Discrete, get_dataloader
        model = Dense_Mnist_Discrete(
                activation=activation,
                initial_weight_type=init_weight_type,
                real_weights_initializer=real_weights_initializer,
                regularize_weights_shayer=reg_logits_l2,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.2, 0.4, 0.4] if dropout_rate is None else dropout_rate,
                enable_local_reparameterization=enable_local_reparameterization,
                enable_local_reparameterization_output=enable_local_reparameterization_output,
                enable_gumbel_straight_through=enable_gumbel_straight_through)
        dummy_input = np.ones((2,784), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size=batch_size, permutation_invariant=True)
    else:
        raise NotImplementedError('Dataset \'{}\' not implemented'.format(dataset))

    #-----------------------------------------------------------------------------------------------------------------------
    # Create the model
    model(dummy_input, True, compute_prediction_updates=True) # Build the model
    compute_prediction_updates = batchnorm_reestimation != 0 # Determines whether EMA updates of batchnorm should be computed online [==False] or after every epoch

    if init_mode_real == 'file':
        model.load_weights(model_file_init).assert_consumed() # assert_consumed ensures that all parameters are loaded (real and discrete weights have different variables, so this can happen)
        print('{} weights loaded from \'{}\''.format(init_weight_type.capitalize(), model_file_init))

    if init_weight_type == 'real':
        # Convert continuous weights to weight_type
        model.replace_weights(weight_type=weight_type, regularize_shayer=reg_logits_l2, init_mode=init_mode_discrete)
    elif init_weight_type != weight_type:
        raise NotImplementedError('Conversion between discrete weight types not implemented. Cannot convert from \'{}\' to \'{}\''.format(init_weight_type, weight_type))

    print('#' * 80)
    model.summary()
    print('Note: The above order is not necessarily the order in which the individual blocks are executed')
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
    learning_rate_variable = tf.Variable(lr_init, tf.float32)
    learning_rate_logits_variable = tf.Variable(lr_init_logits, tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_variable)
    optimizer_logits = tf.keras.optimizers.Adam(learning_rate=learning_rate_logits_variable)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            if enable_local_reparameterization_output:
                predictions = model(images, True, compute_prediction_updates)
                loss = loss_object(labels, predictions)
            else:
                predictions, out_var = model(images, True, compute_prediction_updates, return_output_variance=True) # predictions contains the softmax of out_mean
                loss = loss_object(labels, predictions) + 0.5 * tf.reduce_sum(out_var * predictions * (1.0 - predictions)) / predictions.shape[0]

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
    if tensorboard_logdir is not None:
        logwriter = tf.summary.create_file_writer(tensorboard_logdir)

    print('Start optimization. #epochs: {}. Initial learning rates: {},{}'.format(n_epochs, learning_rate_variable, learning_rate_logits_variable))

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
                          n_epochs,
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

    stats = {'train_loss'           : [],
             'train_error'          : [],
             'val_loss'             : [],
             'val_error'            : [],
             'test_loss'            : [],
             'test_error'           : [],
             't_elapsed_train'      : [],
             't_elapsed_eval_val'   : [],
             't_elapsed_eval_test'  : [],
             'learning_rate'        : [],
             'learning_rate_logits' : []}

    for epoch in range(n_epochs):
        if epoch > 0 and epoch % lr_schedule_reduce_every == 0:
            learning_rate_variable.assign(learning_rate_variable * lr_schedule_reduce_factor)
            learning_rate_logits_variable.assign(learning_rate_logits_variable * lr_schedule_reduce_factor)
            print('Decreasing learning rate. New learning rates: {},{}'.format(learning_rate_variable, learning_rate_logits_variable))

        t0_train = time()
        for images, labels in train_ds:
            train_step(images, labels)

        if compute_prediction_updates:
            # Compute batchnorm updates after epoch
            if batchnorm_reestimation < 0:
                for images, _ in train_ds:
                    train_prediction_updates(images) # take all training data
            elif batchnorm_reestimation > 0:
                for images, _ in train_ds.take(batchnorm_reestimation): # take only batchnorm_reestimation mini-batches
                    train_prediction_updates(images)
            else:
                assert False
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
        pyval_learning_rate_logits = to_pyscalar(learning_rate_logits_variable)

        template = 'Epoch {:3d}/{:3d}, Loss: {:e}, CE[TR]: {:8.5f}, Loss[VA]: {:e}, CE[VA]: {:8.5f}, Loss[TE]: {:e}, CE[TE]: {:8.5f} [t_elapsed: {:6.2f} seconds]'
        print(template.format(epoch+1,
                              n_epochs,
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
                tf.summary.scalar("learning_rate_logits", pyval_learning_rate_logits, step=epoch+1)
        # Write interesting values to stats
        stats['train_loss'].append(pyval_train_loss)
        stats['train_error'].append(pyval_train_error)
        stats['val_loss'].append(pyval_val_loss)
        stats['val_error'].append(pyval_val_error)
        stats['test_loss'].append(pyval_test_loss)
        stats['test_error'].append(pyval_test_error)
        stats['t_elapsed_train'].append(t_elapsed_train)
        stats['t_elapsed_eval_val'].append(t_elapsed_eval_val)
        stats['t_elapsed_eval_test'].append(t_elapsed_eval_test)
        stats['learning_rate'].append(pyval_learning_rate)
        stats['learning_rate_logits'].append(pyval_learning_rate_logits)

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        if np.argmin(stats['val_error']) == len(stats['val_error']) - 1:
            print('Validation error has improved --> Storing model to \'{}\''.format(model_file_best))
            model.save_weights(model_file_best)

    # Save model
    if model_file_last is not None:
        model.save_weights(model_file_last)
        print('Model stored to \'{}\''.format(model_file_last))

    # Convert stats to numpy
    for key in stats:
        stats[key] = np.asarray(stats[key])

    return stats


def main():
    #-----------------------------------------------------------------------------------------------------------------------
    # Set up experiment
    parser = OptionParser()
    parser.add_option('--taskid', action='store', type='int', dest='taskid', default=1)
    parser.add_option('--experiment-dir', action='store', type='string', dest='experiment_dir', default='')
    parser.add_option('--dataset-file', action='store', type='string', dest='dataset_file', default='')
    parser.add_option('--init-model-file', action='store', type='string', dest='init_model_file', default='')
    parser.add_option('--init-weight-type', action='store', type='string', dest='init_weight_type', default='real')
    parser.add_option('--init-mode-real', action='store', type='string', dest='init_mode_real', default='file')
    parser.add_option('--init-mode-discrete', action='store', type='string', dest='init_mode_discrete', default='default')
    parser.add_option('--dataset', action='store', type='string', dest='dataset', default='')
    parser.add_option('--weight-type', action='store', type='string', dest='weight_type', default='')
    parser.add_option('--activation', action='store', type='string', dest='activation', default='')
    parser.add_option('--enable-local-reparameterization', action='store_true', dest='enable_local_reparameterization', default=False)
    parser.add_option('--enable-local-reparameterization-output', action='store_true', dest='enable_local_reparameterization_output', default=False)
    parser.add_option('--enable-gumbel-straight-through', action='store_true', dest='enable_gumbel_straight_through', default=False)
    parser.add_option('--n-epochs', action='store', type='int', dest='n_epochs', default=300)
    parser.add_option('--batch-size', action='store', type='int', dest='batch_size', default=100)
    parser.add_option('--lr-init', action='store', type='float', dest='lr_init', default=1e-3)
    parser.add_option('--lr-init-logits', action='store', type='float', dest='lr_init_logits', default=1e-2)
    parser.add_option('--lr-schedule-reduce-factor', action='store', type='float', dest='lr_schedule_reduce_factor', default=0.1)
    parser.add_option('--lr-schedule-reduce-every', action='store', type='int', dest='lr_schedule_reduce_every', default=100)
    parser.add_option('--reg-logits-l2', action='store', type='float', dest='reg_logits_l2', default=0.0)
    parser.add_option('--reg-kl', action='store', type='float', dest='reg_kl', default=0.0)
    parser.add_option('--batchnorm-momentum', action='store', type='float', dest='batchnorm_momentum', default=0.9)
    parser.add_option('--batchnorm-reestimation', action='store', type='int', dest='batchnorm_reestimation', default=-1)
    parser.add_option('--dropout-rate', action='store', type='string', dest='dropout_rate_str', default='default')
    parser.add_option('--pool-mode', action='store', type='string', dest='pool_mode', default='max')
    parser.add_option('--n-samples-max-pool-reparameterization', action='store', type='int', dest='n_samples_max_pool_reparameterization', default=0)
    parser.add_option('--enable-safe-conv-variance', action='store_true', dest='enable_safe_conv_variance', default=False)

    options, _ = parser.parse_args()
    taskid = options.taskid
    experiment_dir = options.experiment_dir
    dataset_file = options.dataset_file
    model_file_init = options.init_model_file
    init_weight_type = options.init_weight_type
    init_mode_real = options.init_mode_real
    init_mode_discrete = options.init_mode_discrete
    dataset = options.dataset
    weight_type = options.weight_type
    activation = options.activation
    enable_local_reparameterization = options.enable_local_reparameterization
    enable_local_reparameterization_output = options.enable_local_reparameterization_output
    enable_gumbel_straight_through = options.enable_gumbel_straight_through
    n_epochs = options.n_epochs
    batch_size = options.batch_size
    lr_init = options.lr_init
    lr_init_logits = options.lr_init_logits
    lr_schedule_reduce_factor = options.lr_schedule_reduce_factor
    lr_schedule_reduce_every = options.lr_schedule_reduce_every
    reg_logits_l2 = options.reg_logits_l2
    reg_kl = options.reg_kl
    batchnorm_momentum = options.batchnorm_momentum
    batchnorm_reestimation = options.batchnorm_reestimation
    dropout_rate_str = options.dropout_rate_str
    pool_mode = options.pool_mode
    n_samples_max_pool_reparameterization = options.n_samples_max_pool_reparameterization
    enable_safe_conv_variance = options.enable_safe_conv_variance

    assert taskid is not None and taskid >= 1
    assert n_epochs >= 1
    assert options.experiment_dir != ''
    assert options.dataset_file != ''
    assert options.init_mode_real != 'file' or options.init_model_file != ''
    assert activation != ''

    if dropout_rate_str == 'default':
        dropout_rate = None
    else:
        dropout_rate = list(map(float, dropout_rate_str.split(',')))

    rng = np.random.RandomState(seed=58713682)
    rng_seeds = rng.randint(1, 1e9, size=(1000,))

    if taskid > 1000:
        raise Exception('taskid {} too large (only {} defined)'.format(taskid, 1000))
    else:
        print('taskid: {}'.format(taskid))

    rng_seed = rng_seeds[taskid - 1]
    tf.random.set_seed(rng_seed)

    print('-' * 80)
    print('taskid: {}'.format(taskid))
    print('experiment_dir: \'{}\''.format(experiment_dir))
    print('dataset_file: \'{}\''.format(dataset_file))
    print('model_file_init: \'{}\''.format(model_file_init))
    print('init_weight_type: \'{}\''.format(init_weight_type))
    print('init_mode_real: \'{}\''.format(init_mode_real))
    print('init_mode_discrete: \'{}\''.format(init_mode_discrete))
    print('dataset: \'{}\''.format(dataset))
    print('weight_type: \'{}\''.format(weight_type))
    print('activation: \'{}\''.format(activation))
    print('enable_local_reparameterization: {}'.format(enable_local_reparameterization))
    print('enable_local_reparameterization_output: {}'.format(enable_local_reparameterization_output))
    print('enable_gumbel_straight_through: {}'.format(enable_gumbel_straight_through))
    print('n_epochs: {}'.format(n_epochs))
    print('batch_size: {}'.format(batch_size))
    print('lr_init: {}'.format(lr_init))
    print('lr_init_logits: {}'.format(lr_init_logits))
    print('lr_schedule_reduce_factor: {}'.format(lr_schedule_reduce_factor))
    print('lr_schedule_reduce_every: {}'.format(lr_schedule_reduce_every))
    print('reg_logits_l2: {}'.format(reg_logits_l2))
    print('reg_kl: {}'.format(reg_kl))
    print('batchnorm_momentum: {}'.format(batchnorm_momentum))
    print('batchnorm_reestimation: {}'.format(batchnorm_reestimation))
    print('dropout_rate: {}'.format(dropout_rate))
    print('pool_mode: \'{}\''.format(pool_mode))
    print('n_samples_max_pool_reparameterization: {}'.format(n_samples_max_pool_reparameterization))
    print('enable_safe_conv_variance: {}'.format(enable_safe_conv_variance))
    print('-' * 80)
    print('rng_seed: {}'.format(rng_seed))
    print('-' * 80)

    stats_file = '{}/stats/stats_{:03d}.npz'.format(experiment_dir, taskid)
    tensorboard_logdir = '{}/tensorboard/experiment{:03d}'.format(experiment_dir, taskid)
    model_file_best = '{}/models/model_best_{}_{:03d}'.format(experiment_dir, dataset, taskid) # Resulting model will be stored in this file
    model_file_last = '{}/models/model_last_{}_{:03d}'.format(experiment_dir, dataset, taskid) # Resulting model will be stored in this file

    if isfile(stats_file):
        print('Warning. File \'{}\' already exists. We stop here to not overwrite old results.'.format(stats_file))
        exit()

    stats = run(dataset=dataset,
                dataset_file=dataset_file,
                init_weight_type=init_weight_type,
                init_mode_real=init_mode_real,
                init_mode_discrete=init_mode_discrete,
                weight_type=weight_type,
                activation=activation,
                enable_local_reparameterization=enable_local_reparameterization,
                enable_local_reparameterization_output=enable_local_reparameterization_output,
                enable_gumbel_straight_through=enable_gumbel_straight_through,
                n_epochs=n_epochs,
                batch_size=batch_size,
                lr_init=lr_init,
                lr_init_logits=lr_init_logits,
                lr_schedule_reduce_factor=lr_schedule_reduce_factor,
                lr_schedule_reduce_every=lr_schedule_reduce_every,
                reg_logits_l2=reg_logits_l2,
                reg_kl=reg_kl,
                batchnorm_momentum=batchnorm_momentum,
                batchnorm_reestimation=batchnorm_reestimation,
                dropout_rate=dropout_rate,
                pool_mode=pool_mode,
                n_samples_max_pool_reparameterization=n_samples_max_pool_reparameterization,
                enable_safe_conv_variance=enable_safe_conv_variance,
                model_file_init=model_file_init,
                model_file_best=model_file_best,
                model_file_last=model_file_last,
                tensorboard_logdir=tensorboard_logdir)

    stats['experiment_parameters/rng_seed'] = rng_seed

    stats['call_arguments/taskid'] = taskid
    stats['call_arguments/experiment_dir'] = experiment_dir
    stats['call_arguments/dataset_file'] = dataset_file
    stats['call_arguments/model_file_init'] = model_file_init
    stats['call_arguments/init_weight_type'] = init_weight_type
    stats['call_arguments/init_mode_real'] = init_mode_real
    stats['call_arguments/init_mode_discrete'] = init_mode_discrete
    stats['call_arguments/dataset'] = dataset
    stats['call_arguments/weight_type'] = weight_type
    stats['call_arguments/activation'] = activation
    stats['call_arguments/enable_local_reparameterization'] = enable_local_reparameterization
    stats['call_arguments/enable_local_reparameterization_output'] = enable_local_reparameterization_output
    stats['call_arguments/enable_gumbel_straight_through'] = enable_gumbel_straight_through
    stats['call_arguments/n_epochs'] = n_epochs
    stats['call_arguments/batch_size'] = batch_size
    stats['call_arguments/lr_init'] = lr_init
    stats['call_arguments/lr_init_logits'] = lr_init_logits
    stats['call_arguments/lr_schedule_reduce_factor'] = lr_schedule_reduce_factor
    stats['call_arguments/lr_schedule_reduce_every'] = lr_schedule_reduce_every
    stats['call_arguments/reg_logits_l2'] = reg_logits_l2
    stats['call_arguments/reg_kl'] = reg_kl
    stats['call_arguments/batchnorm_momentum'] = batchnorm_momentum
    stats['call_arguments/batchnorm_reestimation'] = batchnorm_reestimation
    stats['call_arguments/dropout_rate'] = dropout_rate_str
    stats['call_arguments/pool_mode'] = pool_mode
    stats['call_arguments/n_samples_max_pool_reparameterization'] = n_samples_max_pool_reparameterization
    stats['call_arguments/enable_safe_conv_variance'] = enable_safe_conv_variance

    np.savez_compressed(stats_file, **stats)
    print('Training statistics stored to \'{}\''.format(stats_file))
    print('Job finished')


if __name__ == '__main__':
    main()
