import sys
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

from metanas.utils.visualize import plot


def plot_training(path, eval_every):
    with (open(path, 'rb')) as f:
        res = pickle.load(f)

    # TODO: Possiblity of combining multiple paths/runs with different
    # seed. Already implemented in notebooks.

    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    # Length like train loss
    test_spacing = np.linspace(0, len(res['train_test_loss']),
                               num=len(res['test_test_loss']),
                               retstep=eval_every, dtype=np.int32)[0]

    axes[0].plot(res['train_test_loss'], 'o-',
                 color="r", label="Training test loss")
    axes[0].plot(test_spacing, res['test_test_loss'],  'o-', color="g",
                 label="Test test loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].legend(loc="best")

    axes[1].plot(res['train_test_accu'], 'o-', color="r",
                 label="Training test accuracy")
    axes[1].plot(test_spacing, res['test_test_accu'], 'o-',
                 color="g", label="Test test accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].legend(loc="best")
    plt.show()

    # Returns the final sparse parameters, accuracy and loss
    return (res['sparse_params_logger'][-1], res['train_test_loss'][-1],
            res['test_test_loss'][-1], res['train_test_accu'][-1],
            res['test_test_accu'][-1])


def plot_genotype(path, eval_every):
    with (open(path, 'rb')) as f:
        res = pickle.load(f)

    for i in range(len(res['genotype'])):
        if i % eval_every == 0:
            plot(res['genotype'][i].normal, 'normal', 'normal cell')
            plot(res['genotype'][i].reduce, 'reduce', 'reduce cell')


if __name__ == '__main__':
    plt.style.use('ggplot')

    parser = argparse.ArgumentParser()
    # TODO: Extract this from the config
    parser.add_argument("--eval_every_n", type=int, default=5,
                        help="Evaluate rate of the experiment")
    parser.add_argument("--path", default="/home/elt4hi/")
    args = parser.parse_args()

    results = plot_training(args.path, args.eval_every_n)
    plot_genotype(args.path, args.eval_every_n)

    print(f"The final number of parameters of {results[0]}")
    print(f"The final training test loss {results[1]}")
    print(f"The final test test loss {results[2]}")
    print(f"The final training test accuracy {results[3]}")
    print(f"The final test test accuracy {results[4]}")
