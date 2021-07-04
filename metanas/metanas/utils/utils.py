
import datetime
import logging
import os
import shutil
import tempfile
import numpy as np
import torch

""" Utilities
Copyright (c) 2021 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

"""


"""
Based on https://github.com/khanrc/pt.darts
which is licensed under MIT License,
cf. 3rd-party-licenses.txt in root directory.
"""


def set_hyperparameter(config):
    """Load/set hyperparameter settings based on predefined config"""

    # Default P-DARTS settings
    # 3 stages as defined in P-DARTS, 5.1.1, keep configuration the same as
    # DARTS in the initial stage.
    config.architecture_stages = 3
    # These are the epochs of all stages combined,
    config.total_meta_epochs = config.architecture_stages * config.meta_epochs

    # The number of operations preserved on each edge of the super-network are,
    # 8, 5, and 3 for stage 1, 2 and 3, respectively.
    # In our case, the third stage will never drop operations.
    config.drop_number_operations = [2, 2, 0]

    # Dropout rate on the operations
    config.dropout_ops = [0, 0.3, 0.7]
    config.dropout_scale_factor = 0.2

    if config.hp_setting == "in_metanas":  # setting for MetaNAS
        config.task_train_steps = 5
        config.n_train = 15
        config.batch_size = 20
        config.batch_size_test = 5 if config.n == 1 else 15  # reptile paper A1
        config.meta_batch_size = 10
        config.w_lr = 0.001
        config.alpha_lr = 0.001
        config.w_meta_lr = 1.0
        config.a_meta_lr = 0.6
        config.a_meta_anneal = 0
        config.a_task_anneal = 0
        config.w_meta_anneal = 0
        config.w_task_anneal = 0

    elif config.hp_setting == "og_metanas":  # setting for MetaNAS
        config.task_train_steps = 5
        config.n_train = 15
        config.batch_size = 20
        config.batch_size_test = 10
        config.meta_batch_size = 10
        config.w_lr = 0.005
        config.alpha_lr = 0.005
        config.w_meta_lr = 1.0
        config.a_meta_lr = 0.6
        config.a_meta_anneal = 0
        config.a_task_anneal = 0
        config.w_meta_anneal = 0
        config.w_task_anneal = 0

    # Settings for MetaNAS with P-DARTS addition
    elif config.hp_setting == "og_pdarts":
        config.task_train_steps = 5
        config.n_train = 15
        config.batch_size = 40
        config.batch_size_test = 10
        config.meta_batch_size = 10
        config.w_lr = 0.005
        config.alpha_lr = 0.005
        config.w_meta_lr = 1.0
        config.a_meta_lr = 0.6
        config.a_meta_anneal = 0
        config.a_task_anneal = 0
        config.w_meta_anneal = 0
        config.w_task_anneal = 0

    elif config.hp_setting == "og":  # default setting from REPTILE paper
        print("Using 'og' hp setting")
        config.task_train_steps = 10
        config.n_train = 10
        config.batch_size = 20
        config.batch_size_test = 10
        config.meta_batch_size = 5
        config.w_lr = 0.0005
        config.alpha_lr = 0.0
        config.w_meta_lr = 1.0
        config.a_meta_lr = 0.0
        config.a_meta_anneal = 0
        config.a_task_anneal = 0
        config.w_meta_anneal = 1
        config.w_task_anneal = 0

    elif config.hp_setting == "in":  # default setting from REPTILE paper
        print("Using 'in' hp setting")
        config.task_train_steps = 8
        config.n_train = 15
        config.batch_size = 10
        config.batch_size_test = 5 if config.n == 1 else 15  # reptile paper A1
        config.meta_batch_size = 5
        config.w_lr = 0.001
        config.alpha_lr = 0.0
        config.w_meta_lr = 1.0
        config.a_meta_lr = 0.0
        config.a_meta_anneal = 0
        config.a_task_anneal = 0
        config.w_meta_anneal = 1
        config.w_task_anneal = 0

    elif config.hp_setting == "test_exp":  # setting for debugging MetaNAS
        config.task_train_steps = 2
        config.n_train = 1
        config.batch_size = 1
        config.batch_size_test = 2
        config.meta_batch_size = 2
        config.w_lr = 0.05
        config.alpha_lr = 0.05
        config.w_meta_lr = 1.0
        config.a_meta_lr = 0.6
        config.a_meta_anneal = 0
        config.a_task_anneal = 0
        config.w_meta_anneal = 0
        config.w_task_anneal = 0

    else:
        raise RuntimeError(f"Unrecognized hp_setting {config.hp_setting}")

    # compatibility with older versions
    if not hasattr(config, "batch_size_test"):
        config.batch_size_test = config.batch_size
    return config


def get_logger(file_path):
    """ Make python logger """
    logger = logging.getLogger("darts")
    log_format = "%(asctime)s | %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%m/%d %I:%M:%S %p")
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def get_experiment_path(config):
    # Write experiment output (logging, parameters, tensorboardX, ..) to
    # experiments/<EXPERIMENT_GROUP>/<DATE>_<NAME>_<UNIQUE_ID>/
    current_date = datetime.datetime.today().strftime(f"%m-%d")
    current_time = datetime.datetime.now().time()
    experiment_group_dir = os.path.join("experiments", config.experiment_group)
    os.makedirs(experiment_group_dir, exist_ok=True)
    experiment_name = f"{current_date}_{config.name}_"

    if config.job_id:
        experiment_name = f"{experiment_name}{config.job_id}"
        experiment_path = os.path.join(experiment_group_dir, experiment_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        else:
            print("----------------------------------")
            print("Warning: Dir already exists. Will overwrite.")
            print("----------------------------------")
    else:
        experiment_path = tempfile.mkdtemp(
            prefix=experiment_name, dir=experiment_group_dir
        )
    return experiment_path


def parse_gpus(gpus):
    if gpus == "all":
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(",")]


def print_config_params(config, prtf=print):
    prtf("")
    prtf("Parameters:")
    for attr, value in sorted(vars(config).items()):
        prtf(f"{attr.upper()}={value}")
    prtf("")


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size())
        for k, v in model.named_parameters()
        if not k.startswith("aux_head")
    )
    return n_params / 1024.0 / 1024.0


class AverageMeter:
    """ Computes and stores the average and current value """

    def __init__(self):
        self.val, self.avg, self.sum, self.count = (0.0, 0.0, 0.0, 0.0)
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EMAMeter:
    """Computes and stores an exponential moving average

    Attributes:
        avg: The current EMA
        alpha: The degree of weight decrease (a higher alpha discounts older
            observations faster)
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.avg = 0.0

    def update(self, val):
        self.avg = self.alpha * val + (1 - self.alpha) * self.avg

    def reset(self):
        self.avg = 0.0


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # Originally, view(-1) is used to reshape,
        # however this caused problems for two dimensional
        # array e.g. (5, 20)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, "best.pth.tar")
        shutil.copyfile(filename, best_filename)


def count_params(net):
    return sum(p.numel() for p in net.parameters())


def save_state(
    meta_model,
    meta_optimizer,
    task_optimizer,
    path: str,
    epoch: int = None,
    job_id: str = None,
):
    """Save the model and optimizer states using pytorch

    Args:
        meta_model:
        meta_optimizer:
        task_optimizer:
        path: The path where the model is stored
        epoch: Epoch that is appended to the file name "meta_state".
        job_id: String that is used to save a temporary file with the
            job_id appended to the state file name. This file is rename
            after saving to the regular name.
    """

    epochpath = os.path.join(path, f"e{epoch}_") if epoch is not None else path

    # save the model (to temporary path if job_id is specified then then
    # rename)
    model_file = epochpath + "meta_state"
    model_file_tmp = model_file if job_id is None else model_file + \
        f"_{job_id}"
    torch.save(
        {
            "meta_model": meta_model.state_dict(),
            "w_meta_optim": meta_optimizer.w_meta_optim.state_dict(),
            "a_meta_optim": meta_optimizer.a_meta_optim.state_dict(),
            "w_task_optim": task_optimizer.w_optim.state_dict(),
            "a_task_optim": task_optimizer.a_optim.state_dict(),
        },
        model_file_tmp,
    )
    if model_file_tmp != model_file:
        os.rename(model_file_tmp, model_file)


def load_state(
    meta_model,
    meta_optimizer,
    task_optimizer,
    path,
    filename="meta_state",
):

    meta_state = torch.load(os.path.join(path, filename))
    meta_model.load_state_dict(meta_state["meta_model"])
    if meta_optimizer is not None:
        meta_optimizer.w_meta_optim.load_state_dict(meta_state["w_meta_optim"])
        meta_optimizer.a_meta_optim.load_state_dict(meta_state["a_meta_optim"])
    if task_optimizer is not None:
        task_optimizer.w_optim.load_state_dict(meta_state["w_task_optim"])
        task_optimizer.a_optim.load_state_dict(meta_state["a_task_optim"])


def load_model_from_state(meta_model, path, strict, filename="meta_state"):
    meta_state = torch.load(os.path.join(path, filename))
    meta_model.load_state_dict(meta_state["meta_model"], strict=strict)


def get_genotype_from_model_ckpt(path, model_instance):
    meta_state = torch.load(path)
    model_instance.load_state_dict(meta_state["meta_model"])
    return model_instance.genotype()
