import os
import igraph
import argparse

import torch

from metanas.meta_predictor.loader import MetaTestDataset
from metanas.meta_predictor.meta_predictor import MetaPredictor
from metanas.utils import utils


def decode_metad2a_to_igraph(row):
    if isinstance(row, str):
        row = eval(row)  # convert string to list of lists
    n = len(row)

    g = igraph.Graph(directed=True)
    g.add_vertices(n)

    for i, node in enumerate(row):
        g.vs[i]['type'] = node[0]

        if i < (n - 2) and i > 0:
            g.add_edge(i, i + 1)  # always connect from last node
        for j, edge in enumerate(node[1:]):
            if edge == 1:
                g.add_edge(j, i)
    return g, n


def main(config):
    meta_pred = MetaPredictor(config)
    meta_pred.meta_train()

    # dataset = MetaTestDataset(
    #     config.data_path,
    #     'mnist',
    #     config.num_samples,
    #     config.num_class)

    # m = "[[0], [6, 1], [4, 1, 0], [3, 0, 1, 0], [4, 1, 0, 0, 0], [3, 0, 1, 0, 0, 0], [3, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 1, 1, 1]]"
    # graph, _ = decode_metad2a_to_igraph(m)

    # y_pred = meta_pred.evaluate_architecture(dataset[0], graph)
    # print(y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help="gpu device ids separated by comma. " "`all` indicates use all"
        " gpus.",
    )
    parser.add_argument("--meta_test", action="store_true",
                        help="Whether in meta-testing stage")
    parser.add_argument('--model_path', default=None,
                        help='select model')
    parser.add_argument('--data_path')
    parser.add_argument('--save_path')

    # Training parameters
    parser.add_argument('--lr', default=1e-4,
                        help='the learning rate of the predictor')
    parser.add_argument('--save_epoch', type=int, default=20,
                        help='save model every n epochs')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size for generator')

    parser.add_argument('--graph_data_name',
                        default='nasbench201', help='graph dataset name')
    parser.add_argument(
        '--nvt',
        type=int,
        default=7,
        help="number of different node types "
        "7: NAS-Bench-201 including in/out node",
    )

    # Set encoder
    parser.add_argument('--num_samples', type=int, default=20,
                        help='the number of images as input for set encoder')
    parser.add_argument('--num_class', type=int, default=5,
                        help='the number of class of dataset')

    # Graph encoder
    parser.add_argument('--hs', type=int, default=56,
                        help='hidden size of GRUs')
    parser.add_argument('--nz', type=int, default=56,
                        help='the number of dimensions of latent vectors z')
    args = parser.parse_args()

    # Set up device
    args.gpus = utils.parse_gpus(args.gpus)
    args.device = torch.device("cuda")

    # Logging
    logger = utils.get_logger(os.path.join(args.path, f"{args.name}.log"))
    args.logger = logger

    main(args)
