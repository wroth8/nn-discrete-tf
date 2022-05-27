import numpy as np
import tensorflow as tf

from optparse import OptionParser
from os.path import isfile
from scipy.special import softmax
from time import time


def run(dataset,
        dataset_file,
        n_max_samples,
        n_experiments,
        weight_type,
        activation,
        batch_size,
        batchnorm_momentum,
        batchnorm_reestimation,
        dropout_rate,
        pool_mode,
        enable_output_activation_normalization,
        init_model_file,
        tensorboard_logdir):
    #-----------------------------------------------------------------------------------------------------------------------
    # Note: For init_weight_type `real` we initialize with real-valued weights (typically from a stored model).
    # These weights will later be replaced by the given weight_type.

    if dataset == 'cifar10':
        from dataset_cifar10 import Vgg32x32_Discrete, get_dataloader
        model = Vgg32x32_Discrete(
                activation=activation,
                initial_weight_type=weight_type,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0] if dropout_rate is None else dropout_rate,
                pool_mode=pool_mode,
                enable_sampled_weights=True,
                enable_output_activation_normalization=enable_output_activation_normalization)
        dummy_input = np.ones((2,32,32,3), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size_eval=batch_size, training_mode=False)
        n_classes = 10
    elif dataset == 'cifar100':
        from dataset_cifar100 import Vgg32x32_Discrete, get_dataloader
        model = Vgg32x32_Discrete(
                activation=activation,
                initial_weight_type=weight_type,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0] if dropout_rate is None else dropout_rate,
                pool_mode=pool_mode,
                enable_sampled_weights=True,
                enable_output_activation_normalization=enable_output_activation_normalization)
        dummy_input = np.ones((2,32,32,3), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size_eval=batch_size, training_mode=False)
        n_classes = 100
    elif dataset == 'svhn':
        from dataset_svhn import Vgg32x32_Discrete, get_dataloader
        model = Vgg32x32_Discrete(
                activation=activation,
                initial_weight_type=weight_type,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.0, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.0] if dropout_rate is None else dropout_rate,
                pool_mode=pool_mode,
                enable_sampled_weights=True,
                enable_output_activation_normalization=enable_output_activation_normalization)
        dummy_input = np.ones((2,32,32,3), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size_eval=batch_size, training_mode=False)
        n_classes = 10
    elif dataset == 'mnist':
        from dataset_mnist import Cnn_Mnist_Discrete, get_dataloader
        model = Cnn_Mnist_Discrete(
                activation=activation,
                initial_weight_type=weight_type,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.0, 0.2, 0.3, 0.0] if dropout_rate is None else dropout_rate,
                conv_kernel_size=(5,5),
                pool_mode=pool_mode,
                enable_sampled_weights=True,
                enable_output_activation_normalization=enable_output_activation_normalization)
        dummy_input = np.ones((2,28,28,1), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size_eval=batch_size, permutation_invariant=False, training_mode=False)
        n_classes = 10
    elif dataset == 'mnist-pi':
        from dataset_mnist import Dense_Mnist_Discrete, get_dataloader
        model = Dense_Mnist_Discrete(
                activation=activation,
                initial_weight_type=weight_type,
                batchnorm_momentum=batchnorm_momentum, # We used 0.9 in the Theano implementation
                dropout_rate=[0.2, 0.4, 0.4] if dropout_rate is None else dropout_rate,
                enable_sampled_weights=True,
                enable_output_activation_normalization=enable_output_activation_normalization)
        dummy_input = np.ones((2,784), dtype=np.float32)
        train_ds, val_ds, test_ds = get_dataloader(dataset_file, batch_size_eval=batch_size, permutation_invariant=True, training_mode=False)
        n_classes = 10
    else:
        raise NotImplementedError('Dataset \'{}\' not implemented'.format(dataset))

    #-----------------------------------------------------------------------------------------------------------------------
    # Create the model
    model(dummy_input, True, compute_prediction_updates=True) # Build the model

    print('#' * 80)
    model.summary()
    print('Note: The above order is not necessarily the order in which the individual blocks are executed')
    print('#' * 80)

    @tf.function
    def predict_sample(images):
        return model(images, training=False, enable_output_softmax=False, use_sampled_weights=True)

    @tf.function
    def predict_most_probable(images):
        return model(images, training=False)

    @tf.function
    def train_prediction_updates(images):
        model(images, training=False, compute_prediction_updates=True, use_sampled_weights=True)

    # Collect labels
    labels_train, labels_val, labels_test = [], [], []
    for _, labels in train_ds:
        labels_train.append(labels.numpy().flatten())
    for _, labels in val_ds:
        labels_val.append(labels.numpy().flatten())
    for _, labels in test_ds:
        labels_test.append(labels.numpy().flatten())
    labels_train = np.concatenate(labels_train, axis=0)
    labels_val = np.concatenate(labels_val, axis=0)
    labels_test = np.concatenate(labels_test, axis=0)
    n_train = labels_train.shape[0]
    n_val = labels_val.shape[0]
    n_test = labels_test.shape[0]
    print('n_train = {}'.format(n_train))
    print('n_val   = {}'.format(n_val))
    print('n_test  = {}'.format(n_test))

    # Load model parameters
    model.load_weights(init_model_file).assert_nontrivial_match() # something must be loaded. we cannot use "assert_consumed" because the sampled weights are not saved
    print('{} weights loaded from \'{}\''.format(weight_type.capitalize(), init_model_file))

    # Compute errors with most probable weights (for no particular reason; just to have it in the log file)
    pred_train = np.zeros((n_train, n_classes), np.float32)
    pred_val = np.zeros((n_val, n_classes), np.float32)
    pred_test = np.zeros((n_test, n_classes), np.float32)
    idx = 0
    for images, _ in train_ds:
        pred_train[idx:(idx+images.shape[0]), :] = pred_train[idx:(idx+images.shape[0]), :] + predict_most_probable(images).numpy()
        idx += images.shape[0]
    idx = 0
    for images, _ in val_ds:
        pred_val[idx:(idx+images.shape[0]), :] = pred_val[idx:(idx+images.shape[0]), :] + predict_most_probable(images).numpy()
        idx += images.shape[0]
    idx = 0
    for images, _ in test_ds:
        pred_test[idx:(idx+images.shape[0]), :] = pred_test[idx:(idx+images.shape[0]), :] + predict_most_probable(images).numpy()
        idx += images.shape[0]
    err_train = np.mean(np.argmax(pred_train, axis=1) != labels_train).item()
    err_val = np.mean(np.argmax(pred_val, axis=1) != labels_val).item()
    err_test = np.mean(np.argmax(pred_test, axis=1) != labels_test).item()
    print('-' * 80)
    print('Most probable weights, CE[TR]: {:8.5f}, CE[VA]: {:8.5f}, CE[TE]: {:8.5f}'.format(err_train, err_val, err_test))

    stats = {'train_error_logits'        : np.zeros((n_experiments, n_max_samples), np.float32),
             'train_error_softmax'       : np.zeros((n_experiments, n_max_samples), np.float32),
             'train_error_majority_vote' : np.zeros((n_experiments, n_max_samples), np.float32),
             'val_error_logits'          : np.zeros((n_experiments, n_max_samples), np.float32),
             'val_error_softmax'         : np.zeros((n_experiments, n_max_samples), np.float32),
             'val_error_majority_vote'   : np.zeros((n_experiments, n_max_samples), np.float32),
             'test_error_logits'         : np.zeros((n_experiments, n_max_samples), np.float32),
             'test_error_softmax'        : np.zeros((n_experiments, n_max_samples), np.float32),
             'test_error_majority_vote'  : np.zeros((n_experiments, n_max_samples), np.float32),
             't_elapsed_sample_weights'  : np.zeros((n_experiments, n_max_samples), np.float32),
             't_elapsed_eval_train'      : np.zeros((n_experiments, n_max_samples), np.float32),
             't_elapsed_eval_val'        : np.zeros((n_experiments, n_max_samples), np.float32),
             't_elapsed_eval_test'       : np.zeros((n_experiments, n_max_samples), np.float32),
             'train_error_logits_bnorm'         : np.zeros((n_experiments, n_max_samples), np.float32),
             'train_error_softmax_bnorm'        : np.zeros((n_experiments, n_max_samples), np.float32),
             'train_error_majority_vote_bnorm'  : np.zeros((n_experiments, n_max_samples), np.float32),
             'val_error_logits_bnorm'           : np.zeros((n_experiments, n_max_samples), np.float32),
             'val_error_softmax_bnorm'          : np.zeros((n_experiments, n_max_samples), np.float32),
             'val_error_majority_vote_bnorm'    : np.zeros((n_experiments, n_max_samples), np.float32),
             'test_error_logits_bnorm'          : np.zeros((n_experiments, n_max_samples), np.float32),
             'test_error_softmax_bnorm'         : np.zeros((n_experiments, n_max_samples), np.float32),
             'test_error_majority_vote_bnorm'   : np.zeros((n_experiments, n_max_samples), np.float32),
             't_elapsed_batchnorm_reestimation' : np.zeros((n_experiments, n_max_samples), np.float32),
             't_elapsed_eval_train_bnorm'       : np.zeros((n_experiments, n_max_samples), np.float32),
             't_elapsed_eval_val_bnorm'         : np.zeros((n_experiments, n_max_samples), np.float32),
             't_elapsed_eval_test_bnorm'        : np.zeros((n_experiments, n_max_samples), np.float32)
             }

    def compute_predictions(images):
        pred_logits = predict_sample(images).numpy()
        pred_softmax = softmax(pred_logits, axis=1)
        pred_majority_vote = np.zeros_like(pred_logits)
        pred_majority_vote[np.arange(pred_majority_vote.shape[0]), np.argmax(pred_logits, axis=1)] = 1
        return pred_logits, pred_softmax, pred_majority_vote


    for experiment_idx in range(n_experiments):
        pred_train_logits = np.zeros((n_train, n_classes), np.float32)
        pred_train_softmax = np.zeros((n_train, n_classes), np.float32)
        pred_train_majorvote = np.zeros((n_train, n_classes), np.float32)
        pred_val_logits = np.zeros((n_val, n_classes), np.float32)
        pred_val_softmax = np.zeros((n_val, n_classes), np.float32)
        pred_val_majorvote = np.zeros((n_val, n_classes), np.float32)
        pred_test_logits = np.zeros((n_test, n_classes), np.float32)
        pred_test_softmax = np.zeros((n_test, n_classes), np.float32)
        pred_test_majorvote = np.zeros((n_test, n_classes), np.float32)

        pred_train_logits_bnorm = np.zeros((n_train, n_classes), np.float32)
        pred_train_softmax_bnorm = np.zeros((n_train, n_classes), np.float32)
        pred_train_majorvote_bnorm = np.zeros((n_train, n_classes), np.float32)
        pred_val_logits_bnorm = np.zeros((n_val, n_classes), np.float32)
        pred_val_softmax_bnorm = np.zeros((n_val, n_classes), np.float32)
        pred_val_majorvote_bnorm = np.zeros((n_val, n_classes), np.float32)
        pred_test_logits_bnorm = np.zeros((n_test, n_classes), np.float32)
        pred_test_softmax_bnorm = np.zeros((n_test, n_classes), np.float32)
        pred_test_majorvote_bnorm = np.zeros((n_test, n_classes), np.float32)
        print('-' * 80)
        print('Experiment {:3d}/{:d}'.format(experiment_idx + 1, n_experiments))
        print('-' * 80)

        if tensorboard_logdir is not None:
            logwriter = tf.summary.create_file_writer(tensorboard_logdir)

        for sample_idx in range(n_max_samples):
            t0_sample_weights = time()
            model.resample_weights()
            t_elapsed_sample_weights = time() - t0_sample_weights

            t0_eval_train = time()
            idx = 0
            for images, _ in train_ds:
                pred_logits, pred_softmax, pred_majority_vote = compute_predictions(images)
                pred_train_logits[idx:(idx+images.shape[0]), :] = pred_train_logits[idx:(idx+images.shape[0]), :] + pred_logits
                pred_train_softmax[idx:(idx+images.shape[0]), :] = pred_train_softmax[idx:(idx+images.shape[0]), :] + pred_softmax
                pred_train_majorvote[idx:(idx+images.shape[0]), :] = pred_train_majorvote[idx:(idx+images.shape[0]), :] + pred_majority_vote
                idx += images.shape[0]
            err_train_logits = np.mean(np.argmax(pred_train_logits, axis=1) != labels_train).item()
            err_train_softmax = np.mean(np.argmax(pred_train_softmax, axis=1) != labels_train).item()
            err_train_majorvote = np.mean(np.argmax(pred_train_majorvote, axis=1) != labels_train).item()
            t_elapsed_eval_train = time() - t0_eval_train

            t0_eval_val = time()
            idx = 0
            for images, _ in val_ds:
                pred_logits, pred_softmax, pred_majority_vote = compute_predictions(images)
                pred_val_logits[idx:(idx+images.shape[0]), :] = pred_val_logits[idx:(idx+images.shape[0]), :] + pred_logits
                pred_val_softmax[idx:(idx+images.shape[0]), :] = pred_val_softmax[idx:(idx+images.shape[0]), :] + pred_softmax
                pred_val_majorvote[idx:(idx+images.shape[0]), :] = pred_val_majorvote[idx:(idx+images.shape[0]), :] + pred_majority_vote
                idx += images.shape[0]
            err_val_logits = np.mean(np.argmax(pred_val_logits, axis=1) != labels_val).item()
            err_val_softmax = np.mean(np.argmax(pred_val_softmax, axis=1) != labels_val).item()
            err_val_majorvote = np.mean(np.argmax(pred_val_majorvote, axis=1) != labels_val).item()
            t_elapsed_eval_val = time() - t0_eval_val

            t0_eval_test = time()
            idx = 0
            for images, _ in test_ds:
                pred_logits, pred_softmax, pred_majority_vote = compute_predictions(images)
                pred_test_logits[idx:(idx+images.shape[0]), :] = pred_test_logits[idx:(idx+images.shape[0]), :] + pred_logits
                pred_test_softmax[idx:(idx+images.shape[0]), :] = pred_test_softmax[idx:(idx+images.shape[0]), :] + pred_softmax
                pred_test_majorvote[idx:(idx+images.shape[0]), :] = pred_test_majorvote[idx:(idx+images.shape[0]), :] + pred_majority_vote
                idx += images.shape[0]
            err_test_logits = np.mean(np.argmax(pred_test_logits, axis=1) != labels_test).item()
            err_test_softmax = np.mean(np.argmax(pred_test_softmax, axis=1) != labels_test).item()
            err_test_majorvote = np.mean(np.argmax(pred_test_majorvote, axis=1) != labels_test).item()
            t_elapsed_eval_test = time() - t0_eval_test

            template = 'Sample {:5d}/{:d} [t_elapsed: {:6.2f} seconds]'
            print(template.format(sample_idx + 1, n_max_samples, t_elapsed_sample_weights + t_elapsed_eval_train + t_elapsed_eval_val + t_elapsed_eval_test))
            template = '{:20s}: CE[TR]: {:8.5f}, CE[VA]: {:8.5f}, CE[TE]: {:8.5f}'
            print(template.format('Average Logits', err_train_logits, err_val_logits, err_test_logits))
            print(template.format('Average Softmax', err_train_softmax, err_val_softmax, err_test_softmax))
            print(template.format('Majority Vote', err_train_majorvote, err_val_majorvote, err_test_majorvote))

            # ------------------------------------------------------------------
            # Re-estimate batchnorm parameters
            t0_batchnorm_reestimation = time()
            if batchnorm_reestimation <= 0:
                for images, _ in train_ds:
                    train_prediction_updates(images)
            else:
                for images, _ in train_ds.take(batchnorm_reestimation):
                    train_prediction_updates(images)
            t_elapsed_batchnorm_reestimation = time() - t0_batchnorm_reestimation

            t0_eval_train_bnorm = time()
            idx = 0
            for images, _ in train_ds:
                pred_logits, pred_softmax, pred_majority_vote = compute_predictions(images)
                pred_train_logits_bnorm[idx:(idx+images.shape[0]), :] = pred_train_logits_bnorm[idx:(idx+images.shape[0]), :] + pred_logits
                pred_train_softmax_bnorm[idx:(idx+images.shape[0]), :] = pred_train_softmax_bnorm[idx:(idx+images.shape[0]), :] + pred_softmax
                pred_train_majorvote_bnorm[idx:(idx+images.shape[0]), :] = pred_train_majorvote_bnorm[idx:(idx+images.shape[0]), :] + pred_majority_vote
                idx += images.shape[0]
            err_train_logits_bnorm = np.mean(np.argmax(pred_train_logits_bnorm, axis=1) != labels_train).item()
            err_train_softmax_bnorm = np.mean(np.argmax(pred_train_softmax_bnorm, axis=1) != labels_train).item()
            err_train_majorvote_bnorm = np.mean(np.argmax(pred_train_majorvote_bnorm, axis=1) != labels_train).item()
            t_elapsed_eval_train_bnorm = time() - t0_eval_train_bnorm

            t0_eval_val_bnorm = time()
            idx = 0
            for images, _ in val_ds:
                pred_logits, pred_softmax, pred_majority_vote = compute_predictions(images)
                pred_val_logits_bnorm[idx:(idx+images.shape[0]), :] = pred_val_logits_bnorm[idx:(idx+images.shape[0]), :] + pred_logits
                pred_val_softmax_bnorm[idx:(idx+images.shape[0]), :] = pred_val_softmax_bnorm[idx:(idx+images.shape[0]), :] + pred_softmax
                pred_val_majorvote_bnorm[idx:(idx+images.shape[0]), :] = pred_val_majorvote_bnorm[idx:(idx+images.shape[0]), :] + pred_majority_vote
                idx += images.shape[0]
            err_val_logits_bnorm = np.mean(np.argmax(pred_val_logits_bnorm, axis=1) != labels_val).item()
            err_val_softmax_bnorm = np.mean(np.argmax(pred_val_softmax_bnorm, axis=1) != labels_val).item()
            err_val_majorvote_bnorm = np.mean(np.argmax(pred_val_majorvote_bnorm, axis=1) != labels_val).item()
            t_elapsed_eval_val_bnorm = time() - t0_eval_val_bnorm

            t0_eval_test_bnorm = time()
            idx = 0
            for images, _ in test_ds:
                pred_logits, pred_softmax, pred_majority_vote = compute_predictions(images)
                pred_test_logits_bnorm[idx:(idx+images.shape[0]), :] = pred_test_logits_bnorm[idx:(idx+images.shape[0]), :] + pred_logits
                pred_test_softmax_bnorm[idx:(idx+images.shape[0]), :] = pred_test_softmax_bnorm[idx:(idx+images.shape[0]), :] + pred_softmax
                pred_test_majorvote_bnorm[idx:(idx+images.shape[0]), :] = pred_test_majorvote_bnorm[idx:(idx+images.shape[0]), :] + pred_majority_vote
                idx += images.shape[0]
            err_test_logits_bnorm = np.mean(np.argmax(pred_test_logits_bnorm, axis=1) != labels_test).item()
            err_test_softmax_bnorm = np.mean(np.argmax(pred_test_softmax_bnorm, axis=1) != labels_test).item()
            err_test_majorvote_bnorm = np.mean(np.argmax(pred_test_majorvote_bnorm, axis=1) != labels_test).item()
            t_elapsed_eval_test_bnorm = time() - t0_eval_test_bnorm

            template = '> with batchnorm parameter reestimation [t_elapsed: {:6.2f} seconds]'
            print(template.format(t_elapsed_batchnorm_reestimation + t_elapsed_eval_train_bnorm + t_elapsed_eval_val_bnorm + t_elapsed_eval_test_bnorm))
            template = '{:20s}: CE[TR]: {:8.5f}, CE[VA]: {:8.5f}, CE[TE]: {:8.5f}'
            print(template.format('Average Logits', err_train_logits_bnorm, err_val_logits_bnorm, err_test_logits_bnorm))
            print(template.format('Average Softmax', err_train_softmax_bnorm, err_val_softmax_bnorm, err_test_softmax_bnorm))
            print(template.format('Majority Vote', err_train_majorvote_bnorm, err_val_majorvote_bnorm, err_test_majorvote_bnorm))

            # ------------------------------------------------------------------
            # Load the model to restore the batchnorm parameter estimates
            model.load_weights(init_model_file).assert_nontrivial_match()

            # Write interesting values to tensorboard
            if tensorboard_logdir is not None:
                with logwriter.as_default():
                    tf.summary.scalar("train_error_logits", err_train_logits * 100.0, step=sample_idx+1)
                    tf.summary.scalar("train_error_softmax", err_train_softmax * 100.0, step=sample_idx+1)
                    tf.summary.scalar("train_error_majority_vote", err_train_majorvote * 100.0, step=sample_idx+1)
                    tf.summary.scalar("val_error_logits", err_val_logits * 100.0, step=sample_idx+1)
                    tf.summary.scalar("val_error_softmax", err_val_softmax * 100.0, step=sample_idx+1)
                    tf.summary.scalar("val_error_majority_vote", err_val_majorvote * 100.0, step=sample_idx+1)
                    tf.summary.scalar("test_error_logits", err_test_logits * 100.0, step=sample_idx+1)
                    tf.summary.scalar("test_error_softmax", err_test_softmax * 100.0, step=sample_idx+1)
                    tf.summary.scalar("test_error_majority_vote", err_test_majorvote * 100.0, step=sample_idx+1)
                    tf.summary.scalar("t_elapsed_sample_weights", t_elapsed_sample_weights, step=sample_idx+1)
                    tf.summary.scalar("t_elapsed_eval_train", t_elapsed_eval_train, step=sample_idx+1)
                    tf.summary.scalar("t_elapsed_eval_val", t_elapsed_eval_val, step=sample_idx+1)
                    tf.summary.scalar("t_elapsed_eval_test", t_elapsed_eval_test, step=sample_idx+1)

                    tf.summary.scalar("train_error_logits_bnorm", err_train_logits_bnorm * 100.0, step=sample_idx+1)
                    tf.summary.scalar("train_error_softmax_bnorm", err_train_softmax_bnorm * 100.0, step=sample_idx+1)
                    tf.summary.scalar("train_error_majority_vote_bnorm", err_train_majorvote_bnorm * 100.0, step=sample_idx+1)
                    tf.summary.scalar("val_error_logits_bnorm", err_val_logits_bnorm * 100.0, step=sample_idx+1)
                    tf.summary.scalar("val_error_softmax_bnorm", err_val_softmax_bnorm * 100.0, step=sample_idx+1)
                    tf.summary.scalar("val_error_majority_vote_bnorm", err_val_majorvote_bnorm * 100.0, step=sample_idx+1)
                    tf.summary.scalar("test_error_logits_bnorm", err_test_logits_bnorm * 100.0, step=sample_idx+1)
                    tf.summary.scalar("test_error_softmax_bnorm", err_test_softmax_bnorm * 100.0, step=sample_idx+1)
                    tf.summary.scalar("test_error_majority_vote_bnorm", err_test_majorvote_bnorm * 100.0, step=sample_idx+1)
                    tf.summary.scalar("t_elapsed_batchnorm_reestimation", t_elapsed_batchnorm_reestimation, step=sample_idx+1)
                    tf.summary.scalar("t_elapsed_eval_train_bnorm", t_elapsed_eval_train_bnorm, step=sample_idx+1)
                    tf.summary.scalar("t_elapsed_eval_val_bnorm", t_elapsed_eval_val_bnorm, step=sample_idx+1)
                    tf.summary.scalar("t_elapsed_eval_test_bnorm", t_elapsed_eval_test_bnorm, step=sample_idx+1)

            # Write interesting values to stats
            stats['train_error_logits'][experiment_idx, sample_idx] = err_train_logits
            stats['train_error_softmax'][experiment_idx, sample_idx] = err_train_softmax
            stats['train_error_majority_vote'][experiment_idx, sample_idx] = err_train_majorvote
            stats['val_error_logits'][experiment_idx, sample_idx] = err_val_logits
            stats['val_error_softmax'][experiment_idx, sample_idx] = err_val_softmax
            stats['val_error_majority_vote'][experiment_idx, sample_idx] = err_val_majorvote
            stats['test_error_logits'][experiment_idx, sample_idx] = err_test_logits
            stats['test_error_softmax'][experiment_idx, sample_idx] = err_test_softmax
            stats['test_error_majority_vote'][experiment_idx, sample_idx] = err_test_majorvote
            stats['t_elapsed_sample_weights'][experiment_idx, sample_idx] = t_elapsed_sample_weights
            stats['t_elapsed_eval_train'][experiment_idx, sample_idx] = t_elapsed_eval_train
            stats['t_elapsed_eval_val'][experiment_idx, sample_idx] = t_elapsed_eval_val
            stats['t_elapsed_eval_test'][experiment_idx, sample_idx] = t_elapsed_eval_test

            stats['train_error_logits_bnorm'][experiment_idx, sample_idx] = err_train_logits_bnorm
            stats['train_error_softmax_bnorm'][experiment_idx, sample_idx] = err_train_softmax_bnorm
            stats['train_error_majority_vote_bnorm'][experiment_idx, sample_idx] = err_train_majorvote_bnorm
            stats['val_error_logits_bnorm'][experiment_idx, sample_idx] = err_val_logits_bnorm
            stats['val_error_softmax_bnorm'][experiment_idx, sample_idx] = err_val_softmax_bnorm
            stats['val_error_majority_vote_bnorm'][experiment_idx, sample_idx] = err_val_majorvote_bnorm
            stats['test_error_logits_bnorm'][experiment_idx, sample_idx] = err_test_logits_bnorm
            stats['test_error_softmax_bnorm'][experiment_idx, sample_idx] = err_test_softmax_bnorm
            stats['test_error_majority_vote_bnorm'][experiment_idx, sample_idx] = err_test_majorvote_bnorm
            stats['t_elapsed_batchnorm_reestimation'][experiment_idx, sample_idx] = t_elapsed_batchnorm_reestimation
            stats['t_elapsed_eval_train_bnorm'][experiment_idx, sample_idx] = t_elapsed_eval_train_bnorm
            stats['t_elapsed_eval_val_bnorm'][experiment_idx, sample_idx] = t_elapsed_eval_val_bnorm
            stats['t_elapsed_eval_test_bnorm'][experiment_idx, sample_idx] = t_elapsed_eval_test_bnorm

    return stats


def main():
    #-----------------------------------------------------------------------------------------------------------------------
    # Set up experiment
    parser = OptionParser()
    parser.add_option('--taskid', action='store', type='int', dest='taskid', default=1)

    parser.add_option('--experiment-dir', action='store', type='string', dest='experiment_dir', default='')
    parser.add_option('--init-model-file', action='store', type='string', dest='init_model_file', default='')
    parser.add_option('--dataset-file', action='store', type='string', dest='dataset_file', default='')
    parser.add_option('--dataset', action='store', type='string', dest='dataset', default='')
    parser.add_option('--n-max-samples', action='store', type='int', dest='n_max_samples', default=100)
    parser.add_option('--n-experiments', action='store', type='int', dest='n_experiments', default=10)
    parser.add_option('--weight-type', action='store', type='string', dest='weight_type', default='')
    parser.add_option('--activation', action='store', type='string', dest='activation', default='')
    parser.add_option('--batch-size', action='store', type='int', dest='batch_size', default=100)
    parser.add_option('--batchnorm-momentum', action='store', type='float', dest='batchnorm_momentum', default=0.9)
    parser.add_option('--batchnorm-reestimation', action='store', type='int', dest='batchnorm_reestimation', default=-1)
    parser.add_option('--dropout-rate', action='store', type='string', dest='dropout_rate_str', default='default')
    parser.add_option('--pool-mode', action='store', type='string', dest='pool_mode', default='max')
    parser.add_option('--enable-output-activation-normalization', action='store_true', dest='enable_output_activation_normalization', default=False)

    options, _ = parser.parse_args()
    taskid = options.taskid
    experiment_dir = options.experiment_dir
    init_model_file = options.init_model_file
    dataset_file = options.dataset_file
    dataset = options.dataset
    n_max_samples = options.n_max_samples
    n_experiments = options.n_experiments
    weight_type = options.weight_type
    activation = options.activation
    batch_size = options.batch_size
    batchnorm_momentum = options.batchnorm_momentum
    batchnorm_reestimation = options.batchnorm_reestimation
    dropout_rate_str = options.dropout_rate_str
    pool_mode = options.pool_mode
    enable_output_activation_normalization = options.enable_output_activation_normalization

    assert taskid is not None and taskid >= 1
    assert options.experiment_dir != ''
    assert options.dataset_file != ''
    assert options.init_model_file != ''
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
    print('init_model_file: \'{}\''.format(init_model_file))
    print('dataset_file: \'{}\''.format(dataset_file))
    print('dataset: \'{}\''.format(dataset))
    print('n_max_samples: {}'.format(n_max_samples))
    print('n_experiments: {}'.format(n_experiments))
    print('weight_type: \'{}\''.format(weight_type))
    print('activation: \'{}\''.format(activation))
    print('batch_size: {}'.format(batch_size))
    print('batchnorm_momentum: {}'.format(batchnorm_momentum))
    print('batchnorm_reestimation: {}'.format(batchnorm_reestimation))
    print('dropout_rate: {}'.format(dropout_rate))
    print('pool_mode: \'{}\''.format(pool_mode))
    print('enable_output_activation_normalization: {}'.format(enable_output_activation_normalization))
    print('-' * 80)
    print('rng_seed: {}'.format(rng_seed))
    print('-' * 80)

    stats_file = '{}/stats/stats_{:03d}.npz'.format(experiment_dir, taskid)
    tensorboard_logdir = '{}/tensorboard/experiment{:03d}'.format(experiment_dir, taskid)
    if isfile(stats_file):
        print('Warning. File \'{}\' already exists. We stop here to not overwrite old results.'.format(stats_file))
        exit()

    stats = run(dataset=dataset,
                dataset_file=dataset_file,
                n_max_samples=n_max_samples,
                n_experiments=n_experiments,
                weight_type=weight_type,
                activation=activation,
                batch_size=batch_size,
                batchnorm_momentum=batchnorm_momentum,
                batchnorm_reestimation=batchnorm_reestimation,
                dropout_rate=dropout_rate,
                pool_mode=pool_mode,
                enable_output_activation_normalization=enable_output_activation_normalization,
                init_model_file=init_model_file,
                tensorboard_logdir=tensorboard_logdir)

    stats['experiment_parameters/rng_seed'] = rng_seed

    stats['call_arguments/taskid'] = taskid
    stats['call_arguments/experiment_dir'] = experiment_dir
    stats['call_arguments/init_model_file'] = init_model_file
    stats['call_arguments/dataset_file'] = dataset_file
    stats['call_arguments/dataset'] = dataset
    stats['call_arguments/n_max_samples'] = n_max_samples
    stats['call_arguments/n_experiments'] = n_experiments
    stats['call_arguments/weight_type'] = weight_type
    stats['call_arguments/activation'] = activation
    stats['call_arguments/batch_size'] = batch_size
    stats['call_arguments/batchnorm_momentum'] = batchnorm_momentum
    stats['call_arguments/batchnorm_reestimation'] = batchnorm_reestimation
    stats['call_arguments/dropout_rate'] = dropout_rate_str
    stats['call_arguments/pool_mode'] = pool_mode
    stats['call_arguments/enable_output_activation_normalization'] = enable_output_activation_normalization

    np.savez_compressed(stats_file, **stats)
    print('Training statistics stored to \'{}\''.format(stats_file))
    print('Job finished')


if __name__ == '__main__':
    main()
