#!/bin/sh

# -----------------------------------------------------------------------------
# Experiments that were used to compute the results of the thesis
#
# @PHDTHESIS{Roth2021,
#   author    = {Wolfgang Roth},
#   title     = {Probabilistic Methods for Resource Efficiency in Machine Learning},
#   school    = {Graz University of Technology},
#   year      = {2021}
# }
#
# Note: An older Theano version of this code was used to compute the results of the paper
#
# @INPROCEEDINGS{Roth2019,
#   author    = {Wolfgang Roth and G{\"{u}}nther Schindler and Holger Fr{\"{o}}ning and Franz Pernkopf},
#   title     = {Training Discrete-Valued Neural Networks with Sign Activations Using Weight Distributions},
#   booktitle = {European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
#   year      = {2019}
# }
#
# -----------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Helper function to create directories
#

create_dirs() {
  mkdir datasets
  mkdir results
  mkdir results/$1
  mkdir results/$1/$2
  mkdir results/$1/$2/stats
  mkdir results/$1/$2/logs
}


# ------------------------------------------------------------------------------
# Set the PYTHONPATH and PYTHON_UNBUFFERED
#

CUR_PWD="$(pwd)"
cd ".."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd "${CUR_PWD}"
echo "${PYTHONPATH}"

export PYTHONUNBUFFERED=TRUE


# ------------------------------------------------------------------------------
# Set the TASK_ID
# Each experiment is called with a task ID. The task ID is appended to the result files of each experiment. This is
# useful for tools such as slurm.
#

TASK_ID="1"
TASK_ID_AUX=$(printf "%03d" ${TASK_ID})  # append leading zeros to match the result model file


# ------------------------------------------------------------------------------
# Perform real-valued pre-training using ReLU activations
#
# Notes:
#  --task-id: The `--taskid` is appended to the result file name (useful for tools like slurm)
#  --experiment-dir: Results will be stored in this directory (make sure this folder contains a "stats" sub-directory)
#  --dataset-file: If not already present, the dataset will be downloaded to this file (make sure the directory exists)
#  --dataset: Supported datasets are "mnist", "mnist-pi", "cifar10", "cifar100", "svhn"
#

EXPERIMENT_DIR=pretrain_relu
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_pretrain_real.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --activation=relu \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-4 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-l2=1e-3 \
        --batchnorm-momentum=0.99 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and use sign activations
#   - Use local reparemeterization trick (at hidden and output layers) with the Gumbel softmax approximation
#
# Notes:
#   --init-model-file: Specify the model file that was produced by the previous training run
#   --weight-type: Supported weight types are "ternary", "ternary_shayer", "quaternary", "quinary"
#

EXPERIMENT_DIR=train_ternary_sign_lrt_gumbel_softmax
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/pretrain_relu/models/model_best_cifar10_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=sign \
        --enable-local-reparameterization \
        --enable-local-reparameterization-output \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform weight sampling and prediction averaging using a trained weight distribution model
#   - Three types of prediction averaging are performed:
#     (1) logit averaging (averaging the values before the softmax is applied)
#     (2) softmax averaging (averaging the output class probabilities)
#     (3) majority vote
#   - Each of these three types is performed in two modes:
#     (1) using sampled weights and trained batch normalization parameters
#     (2) using sampled weights and re-estimated batch normalization parameters for the sampled weights
#   - This results in six different averaging results
#
# Notes:
#   --n-experiments: Determines how many sampling experiments should be performed
#   --n-max-samples: Determines how many predictions should be averaged per sampling experiment
#   --batch-size: The batch size used for both evaluation and re-estimation of the batch normalization parameters
#   --batchnorm-momentum: The momentum for batch normalization re-estimation
#   --batchnorm-reestimation: The number of batches used for batch normalization re-estimation. If set to a value lower
#     or equal to zero, an iteration over the whole training data is used for batch normalization re-estimation.
#

EXPERIMENT_DIR=sampling_ternary_sign_lrt_gumbel_softmax
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_evaluate_sampling.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/sampling_ternary_sign_lrt_gumbel_softmax \
        --init-model-file=results/cifar10/train_ternary_sign_lrt_gumbel_softmax/models/model_best_cifar10_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --n-max-samples=100 \
        --n-experiments=10 \
        --weight-type=ternary \
        --activation=sign \
        --batch-size=100 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and use sign activations
#   - Use local reparemeterization trick (at hidden and output layers) with the Gumbel straight through approximation
#

EXPERIMENT_DIR=train_ternary_sign_lrt_gumbel_straight_through
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/pretrain_relu/models/model_best_cifar10_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=sign \
        --enable-local-reparameterization \
        --enable-local-reparameterization-output \
        --enable-gumbel-straight-through \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and use sign activations
#   - Use probabilistic forward pass, use local reparameterization trick only at the output layer
#

EXPERIMENT_DIR=train_ternary_sign_pfp_lrt_output
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/pretrain_relu/models/model_best_cifar10_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=sign \
        --enable-local-reparameterization-output \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and use sign activations
#   - Use probabilistic forward pass in all layers
#

EXPERIMENT_DIR=train_ternary_sign_pfp
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/pretrain_relu/models/model_best_cifar10_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=sign \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions but keep the ReLU activation
#

EXPERIMENT_DIR=train_ternary_relu_lrt_gumbel_softmax
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/pretrain_relu/models/model_best_cifar10_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=relu \
        --enable-local-reparameterization \
        --enable-local-reparameterization-output \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
        --enable-safe-conv-variance \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and use sign activations
#   - Use pre-trained model with discrete weight distributions and ReLU activations
#

EXPERIMENT_DIR=train_ternary_sign_from_pretrained_ternary_relu
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/train_ternary_relu_lrt_gumbel_softmax/models/model_best_cifar10_${TASK_ID_AUX} \
        --init-weight-type=ternary \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=sign \
        --enable-local-reparameterization \
        --enable-local-reparameterization-output \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and sign activations
#   - Use different dropout rates
#

EXPERIMENT_DIR=train_ternary_sign_lrt_gumbel_softmax_different_dropout
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/pretrain_relu/models/model_best_cifar10_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=sign \
        --enable-local-reparameterization \
        --enable-local-reparameterization-output \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
        --dropout-rate=0.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and sign activations
#   - Do not compute batch normalization statistics with the prediction path containing discrete weights and
#     activations. Instead compute exponential moving average statistics with activations as they appear during
#     training. Note that this should perform worse.
#

EXPERIMENT_DIR=train_ternary_sign_lrt_gumbel_softmax_batchnorm_ema
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/pretrain_relu/models/model_best_cifar10_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=sign \
        --enable-local-reparameterization \
        --enable-local-reparameterization-output \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=0 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and sign activations
#   - Use different pooling modes
#
# Notes:
#   --pool-mode: Supported pool modes are "max", "max_sign", "max_mean", "max_sample", "max_shekhovtsov", "max_shekhovtsov_fast", "max_sample_reparam"
#   --n-samples-max-pool-reparameterization: Specify the number of drawn samples for pool mode "max_sample_reparam"
#

EXPERIMENT_DIR=train_ternary_sign_lrt_gumbel_softmax_different_pool_mode
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/pretrain_relu/models/model_best_cifar10_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=sign \
        --enable-local-reparameterization \
        --enable-local-reparameterization-output \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
        --pool-mode=max_shekhovtsov \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and sign activations
#   - Use Shayer parameterization
#
# Notes:
#   --init-mode-discrete: Use "roth" to use the Shayer parameterization with the Roth initialization
#

EXPERIMENT_DIR=train_ternary_sign_lrt_gumbel_softmax_shayer_parameterization
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-model-file=results/cifar10/pretrain_relu/models/model_best_cifar10_${TASK_ID_AUX} \
        --init-mode-discrete=shayer \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary_shayer \
        --activation=sign \
        --enable-local-reparameterization \
        --enable-local-reparameterization-output \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and sign activations
#    - Use different methods for the initialization of the weight distributions
#
# Notes:
#   --init-mode-real: Allows setting different (pseudo) pre-trained weights or, more precisely, the initialization mode
#     of real-valued weights that are then treated as if they come from a pre-trained model to initialize the weight
#     distributions. The default value "file" actually enables loading pre-trained values from a file. Other supported
#     values of `--init-mode-real` are "normal", "uniform", "equidist", and "equidist_ternary".
#   --init-mode-discrete: Allows setting different methods for initialization of the weight distributions for ternary
#     weights. For weights of type "ternary" and "ternary_shayer", supported values of `--init-mode-discrete` are
#     "default", "roth", "roth_without_normalization", "shayer", and "shayer_without_normalization".
#

EXPERIMENT_DIR=train_ternary_sign_lrt_gumbel_softmax_different_initialization
create_dirs cifar10 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar10/$EXPERIMENT_DIR \
        --init-mode-real=normal \
        --init-mode-discrete=roth_without_normalization \
        --dataset-file=datasets/cifar10.npz \
        --dataset=cifar10 \
        --weight-type=ternary \
        --activation=sign \
        --enable-local-reparameterization \
        --enable-local-reparameterization-output \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=1e-3 \
        --lr-init-logits=1e-2 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-logits-l2=1e-10 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar10/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err
