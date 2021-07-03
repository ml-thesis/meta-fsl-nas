import os
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.gradient_based import gradient_update_parameters

from model import ConvolutionalNeuralNetwork
from utils import get_accuracy

logger = logging.getLogger(__name__)


def train(args):
    dataset = omniglot(args.folder,
                       shots=args.num_shots,
                       ways=args.num_ways,
                       shuffle=True,
                       test_shots=15,
                       meta_train=True,
                       download=args.download)
    dataloader = BatchMetaDataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=args.num_workers)

    model = ConvolutionalNeuralNetwork(1,
                                       args.num_ways,
                                       hidden_size=args.hidden_size)
    model.to(device=args.device)
    model.train()
    # meta_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    with tqdm(dataloader, total=args.num_batches) as pbar:
        for batch_idx, batch in enumerate(pbar):
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=args.device)
            train_targets = train_targets.to(device=args.device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=args.device)
            test_targets = test_targets.to(device=args.device)

            outer_loss = torch.tensor(0., device=args.device)
            accuracy = torch.tensor(0., device=args.device)

            weights_before = copy.deepcopy(model.state_dict())

            # Inner loop of sampled tasks, T_i
            # for k_i in range(k): inner epochs
            for i in range(args.inner_steps):
                for task_idx, (train_input, train_target, test_input,
                               test_target) in enumerate(zip(train_inputs,
                                                             train_targets,
                                                             test_inputs,
                                                             test_targets)):

                    train_logit = model(train_input)
                    inner_loss = F.cross_entropy(train_logit, train_target)

                    model.zero_grad()
                    inner_loss.backward()

                    # Update model parameters
                    for param in model.parameters():
                        param.data -= args.step_size * param.grad.data

                    test_logit = model(test_input)
                    outer_loss += F.cross_entropy(test_logit,
                                                  test_target).item(
                    )/(args.batch_size * args.inner_steps)

                    with torch.no_grad():
                        accuracy += get_accuracy(test_logit,
                                                 test_target).item(
                        )/(args.batch_size * args.inner_steps)

            # Update the parameters in the meta-learning step
            weights_after = model.state_dict()

            outerstepsize = 0.1 * \
                (1 - batch_idx / (args.inner_steps *
                 args.num_batches))  # linear schedule
            model.load_state_dict({name:
                                   weights_before[name] +
                                   (weights_after[name] -
                                    weights_before[name]) * outerstepsize
                                   for name in weights_before})

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))
            if batch_idx >= args.num_batches:
                break


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Variant of MAML, Reptile')

    parser.add_argument('folder', type=str,
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--inner-steps', type=float, default=5,
                        help='Inner gradient steps for adaptation (default: 5).')
    parser.add_argument('--step-size', type=float, default=0.4,
                        help='Step-size for the gradient step for adaptation (default: 0.4).')
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of tasks in a mini-batch of tasks (default: 10).')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batches the model is trained over (default: 100).')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--download', action='store_true',
                        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
                        help='Use CUDA if available.')

    args = parser.parse_args()
    args.device = torch.device('cuda' if args.use_cuda
                               and torch.cuda.is_available() else 'cpu')

    train(args)
