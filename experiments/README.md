# NN Discrete TF - Experiments

This directory contains code for running experiments in a more systematic manner.

## Dataset Files

Python files starting with `dataset_` contain code for loading the respective datasets and definitions of dataset-specific models that were evaluated in my PhD thesis and in our ECML-2019 paper.

## Experiment Files

Python files starting with `experiment_` can be called with various command line arguments. Typically, the experiment files are called in the following order:

1. `experiment_pretrain_real.py`: Code for pre-training a real-valued neural network.
2. `experiment_discrete.py`: Code for training neural networks with discrete weight distributions and discrete activation functions. Typically, a pre-trained real-valued neural network is provided to initialize the weight distributions.
3. `experiment_evaluate_sampling.py`: Code for sampling many discrete-valued neural networks from trained discrete weight distributions and performing model averaging.

Note: The experiment files can be called with the `--task-id` argument which is convenient for parallel processing using tools like slurm.

## Run Experiments

Files starting with `run_experiments_` illustrate how the experiment files are called with the command line arguments that were also used in my thesis along with a short description of the experiment. Example calls to the three experiment files listed above are provided for every dataset. For the Cifar-10 dataset, we also provide an extended list of example calls that were also used to produce some more results of my thesis.