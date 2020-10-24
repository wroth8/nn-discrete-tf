# NN Discrete TF

This repository essentially contains a Tensorflow implementation of our previous Theano implementation that can be found on

    https://github.com/wroth8/nn-discrete

to reproduce results from our paper

```
@INPROCEEDINGS{Roth2019,
    AUTHOR="Wolfgang Roth and G{\"{u}}nther Schindler and Holger Fr{\"{o}}ning and Franz Pernkopf",
    TITLE="Training Discrete-Valued Neural Networks with Sign Activations Using Weight Distributions",
    BOOKTITLE="European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)",
    YEAR=2019
}
```

## Usage

The usage of our code is best understood by going through the provided example.

1. Run `example_cifar10_real.py` to train a real-valued CNN with ReLU activation function. The model will be stored in the `saved_models` directory. Some training statistics for tensorboard are stored in the `logs` folder.

2. When a real-valued model has finished training, run `example_cifar10_ternary.py`. In this experiment, the pre-trained real-valued weights from (1.) are used to initialize the ternary weight distributions and then a CNN with ternary weights using the sign activation function is trained.
