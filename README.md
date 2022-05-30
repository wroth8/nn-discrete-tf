# NN Discrete TF

This repository contains a TensorFlow implementation of our previous Theano implementation that can be found at

    https://github.com/wroth8/nn-discrete

The TensorFlow implementation was used to produce the results of my PhD thesis

```
@PHDTHESIS{Roth2021,
  author    = {Wolfgang Roth},
  title     = {Probabilistic Methods for Resource Efficiency in Machine Learning},
  school    = {Graz University of Technology},
  year      = {2021}
}
```

The old Theano implementation was used to produce the results of our paper

```
@INPROCEEDINGS{Roth2019,
  author    = {Wolfgang Roth and G{\"{u}}nther Schindler and Holger Fr{\"{o}}ning and Franz Pernkopf},
  title     = {Training Discrete-Valued Neural Networks with Sign Activations Using Weight Distributions},
  booktitle = {European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
  year      = {2019}
}
```

## Setup

1. Clone this repository: `git clone https://github.com/wroth8/nn-discrete-tf.git`
2. Create a virtual conda environment from the included environment.yml and activate it.
    1. Create using conda: `conda env create -f environment.yml`
    2. Activate using conda: `conda activate nn-discrete-tf`

## Usage

The usage of our code is best understood by going through the provided example experiments.

1. Run `example_cifar10_real.py` to train a real-valued CNN with ReLU activation function. The model will be stored in the `saved_models` directory. Some training statistics for tensorboard are stored in the `logs` directory.

2. Once training of a real-valued model has finished, run `example_cifar10_ternary.py`. In this experiment, the pre-trained real-valued weights from (1.) are used to initialize the ternary weight distributions. Then a CNN with ternary weights using the sign activation function is trained.

A more comprehensive and systematic experiment setup can be found in the `experiments` directory.