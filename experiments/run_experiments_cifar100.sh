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
create_dirs cifar100 $EXPERIMENT_DIR
(python experiment_pretrain_real.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar100/$EXPERIMENT_DIR \
        --dataset-file=datasets/cifar100.npz \
        --dataset=cifar100 \
        --activation=relu \
        --n-epochs=300 \
        --batch-size=100 \
        --lr-init=3e-4 \
        --lr-schedule-reduce-factor=0.1 \
        --lr-schedule-reduce-every=100 \
        --reg-l2=1e-3 \
        --batchnorm-momentum=0.99 \
            | tee results/cifar100/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar100/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


# ------------------------------------------------------------------------------
# Perform training of discrete weight distributions and use sign activations
#   - Use local reparemeterization trick (at hidden and output layers) with the Gumbel softmax approximation
#
# Notes:
#   --init-model-file: Specify the model file that was produced by the previous training run
#   --weight-type: Supported weight types are "ternary", "ternary_shayer", "quaternary", "quinary"
#

EXPERIMENT_DIR=train_ternary_sign_lrt_gumbel_softmax
create_dirs cifar100 $EXPERIMENT_DIR
(python experiment_discrete.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar100/$EXPERIMENT_DIR \
        --init-model-file=results/cifar100/pretrain_relu/models/model_best_cifar100_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar100.npz \
        --dataset=cifar100 \
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
            | tee results/cifar100/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar100/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err


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
create_dirs cifar100 $EXPERIMENT_DIR
(python experiment_evaluate_sampling.py \
        --taskid=$TASK_ID \
        --experiment-dir=results/cifar100/sampling_ternary_sign_lrt_gumbel_softmax \
        --init-model-file=results/cifar100/train_ternary_sign_lrt_gumbel_softmax/models/model_best_cifar100_${TASK_ID_AUX} \
        --dataset-file=datasets/cifar100.npz \
        --dataset=cifar100 \
        --n-max-samples=100 \
        --n-experiments=10 \
        --weight-type=ternary \
        --activation=sign \
        --batch-size=100 \
        --batchnorm-momentum=0.9 \
        --batchnorm-reestimation=100 \
            | tee results/cifar100/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.out) 3>&1 1>&2 2>&3 \
            | tee results/cifar100/$EXPERIMENT_DIR/logs/log$TASK_ID_AUX.err
