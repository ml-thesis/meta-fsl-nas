import copy
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict, namedtuple
from metanas.utils import utils
from metanas.models.search_cnn import SearchCNNController
from metanas.utils import genotypes as gt

""" DARTS algorithm
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


class Darts:
    def __init__(self, model, config, do_schedule_lr=False):
        self.config = config
        self.model = model
        self.primitives = config.primitives

        self.do_schedule_lr = do_schedule_lr
        self.task_train_steps = config.task_train_steps
        self.test_task_train_steps = config.test_task_train_steps
        self.warm_up_epochs = config.warm_up_epochs

        # weights optimizer
        self.w_optim = torch.optim.Adam(
            self.model.weights(),
            lr=self.config.w_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.w_weight_decay,
        )  #

        # architecture optimizer
        self.a_optim = torch.optim.Adam(
            self.model.alphas(),
            self.config.alpha_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.alpha_weight_decay,
        )
        self.architect = Architect(
            self.model,
            self.config.w_momentum,
            self.config.w_weight_decay,
            self.config.use_first_order_darts,
        )

    # def reduce_operations(self, config, current_stage,
    #                       alpha_normal, alpha_reduce):
    #     """P-DARTS, Obtain alpha weights to reduce the operations by the
    #     specified amount for the current stage.
    #     TODO: Can we consider the pairwise alphas here as well?
    #     """
    #     switches_normal = copy.deepcopy(config.switches_normal)
    #     switches_reduce = copy.deepcopy(config.switches_reduce)

    #     # Add again if we consider the None operations
    #     # last_stage = current_stage == config.architecture_stages

    #     # Number of operations to drop, -1 for the index
    #     ops_drop = config.drop_number_operations[current_stage-1]

    #     weights_normal = [self.apply_normalizer(
    #         alpha).data.cpu().numpy() for alpha in self.alpha_normal]
    #     weights_reduce = [self.apply_normalizer(
    #         alpha).data.cpu().numpy() for alpha in self.alpha_reduce]

    #     weights_normal = np.concatenate(weights_normal, axis=0)
    #     weights_reduce = np.concatenate(weights_reduce, axis=0)

    #     switches_reduce = self._adjust_switches(weights_reduce,
    #                                             switches_reduce,
    #                                             ops_drop, config.edges)

    #     switches_normal = self._adjust_switches(weights_normal,
    #                                             switches_normal,
    #                                             ops_drop, config.edges)
    #     return switches_normal, switches_reduce

    # def _adjust_switches(self, weights, switches, ops_drop,
    #                      edges):
    #     """Original P-DARTS, There are 4 intermediate nodes in a cell,
    #     resulting in 2 + 3 + 4 + 5 = 14 edges. So 14 indicates the
    #     number of edges in a cell."""

    #     for i in range(edges):
    #         idxs = np.where(switches[i])[0].tolist()

    #         # If None in primitives, add check for this operation
    #         # if last_stage:
    #         #     # for the last stage, drop all Zero operations
    #         #     drop = self._get_min_k_no_zero(
    #         #         weights[i, :], idxs, ops_drop)
    #         # else:

    #         # get minimum k (ops_drop) from the weights
    #         # original code, drop = get_min_k(prob[i, :], ops_drop)
    #         drop = np.array(weights[i, :]).argsort()[:ops_drop]
    #         for idx in drop:
    #             switches[i][idxs[idx]] = False
    #     return switches

    def step(
        self,
        task,
        epoch,
        global_progress="",
        test_phase=False,
        alpha_logger=None,
        sparsify_input_alphas=None,
        num_of_skip_connections=None,
    ):

        log_alphas = False

        # P-DARTS stages
        if self.config.use_search_space_approximation or \
                self.config.use_search_space_regularization:
            stages = self.config.architecture_stages
        else:
            stages = 1

        if test_phase:
            top1_logger = self.config.top1_logger_test
            losses_logger = self.config.losses_logger_test
            train_steps = self.config.test_task_train_steps
            arch_adap_steps = int(
                train_steps * self.config.test_adapt_steps * stages)

            if alpha_logger is not None:
                log_alphas = True

        else:
            top1_logger = self.config.top1_logger
            losses_logger = self.config.losses_logger
            train_steps = self.config.task_train_steps
            arch_adap_steps = train_steps * stages

        lr = self.config.w_lr

        if self.config.w_task_anneal:
            # reset lr to base lr
            for group in self.w_optim.param_groups:
                group["lr"] = self.config.w_lr

            w_task_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.w_optim, train_steps * stages, eta_min=0.0
            )
        else:
            w_task_lr_scheduler = None

        if self.config.a_task_anneal:
            for group in self.a_optim.param_groups:
                group["lr"] = self.config.alpha_lr

            a_task_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.a_optim, arch_adap_steps, eta_min=0.0
            )
        else:
            a_task_lr_scheduler = None

        model_has_normalizer = hasattr(self.model, "normalizer")
        if model_has_normalizer:
            self.model.normalizer["params"]["curr_step"] = 0.0
            self.architect.v_net.normalizer["params"]["curr_step"] = 0.0
            self.model.normalizer["params"]["max_steps"] = float(
                arch_adap_steps)
            self.architect.v_net.normalizer["params"]["max_steps"] = float(
                arch_adap_steps
            )

        if self.config.drop_path_prob > 0.0:
            # do drop path if not test phase (=in train phase) or if also use
            # in test phase
            if not test_phase or self.config.use_drop_path_in_meta_testing:
                self.model.drop_path_prob(self.config.drop_path_prob)

        if self.config.use_search_space_approximation or \
                self.config.use_search_space_regularization:

            # P-DARTS
            # addition of staging (G_k) for Search Space Approximation and
            # Regularization.
            scale_factor = self.config.dropout_scale_factor

            edges = sum(i for i in range(2, self.config.n_nodes+2))
            switches_normal = np.ones((edges,
                                       len(self.primitives)),
                                      dtype=bool).tolist()
            switches_reduce = np.ones((edges,
                                       len(self.primitives)),
                                      dtype=bool).tolist()

            # For softmax temperature
            global_train_steps = 0
            for current_stage in range(self.config.architecture_stages):

                # Number of ops to preserve
                n_ops = len(self.primitives) - \
                    self.config.drop_number_operations[current_stage]
                dropout_stage = self.config.dropout_ops[current_stage]

                if current_stage != 0:
                    # We increase the depth of the super-network by stacking
                    # more cells, i.e., L_k > L_k−1
                    if self.config.use_search_space_approximation and \
                            self.config.use_reinitialize_model:
                        # Pass the config here
                        self.model.reinit_search_model()

                for train_step in range(train_steps):
                    warm_up = (
                        epoch < self.warm_up_epochs
                    )  # if epoch < warm_up_epochs, do warm up

                    # Set the dropout rate for skip-connections,
                    if self.config.use_search_space_regularization:
                        # Exponential decay in dropout rate
                        dropout_rate = dropout_stage * \
                            np.exp(-global_train_steps * scale_factor)

                        if not test_phase:
                            self.model.drop_out_skip_connections(dropout_rate)

                    # tracked based on the global steps not the stage training
                    # steps
                    if (
                        global_train_steps >= arch_adap_steps
                    ):  # no architecture adap after arch_adap_steps steps
                        warm_up = 1

                    if w_task_lr_scheduler is not None:
                        w_task_lr_scheduler.step()

                    if a_task_lr_scheduler is not None:
                        a_task_lr_scheduler.step()

                    train(
                        task,
                        self.model,
                        self.architect,
                        self.w_optim,
                        self.a_optim,
                        lr,
                        global_progress,
                        self.config,
                        warm_up,
                    )

                    if (
                        model_has_normalizer
                        and global_train_steps < (arch_adap_steps - 1)
                        and not warm_up
                    ):  # todo check if not warm_up is correct
                        self.model.normalizer["params"]["curr_step"] += 1
                        self.architect.v_net.normalizer["params"]["curr_step"] += 1

                    global_train_steps += 1

                if not warm_up:
                    # We reduce the operation space of O_k candidate operations
                    # at the end of each stage, i.e. |O^k_(i,j)| = O_k > O_k-1
                    if current_stage+1 != self.config.architecture_stages \
                            or self.config.use_search_space_approximation:

                        with torch.no_grad():

                            for normal, reduce in zip(self.model.alpha_normal,
                                                      self.model.alpha_reduce):
                                _, indices = torch.topk(normal[:, :], n_ops)
                                _, indices = torch.topk(reduce[:, :], n_ops)

                                mask = torch.zeros(len(normal), len(
                                    self.primitives)).cuda()
                                normal.data = mask.scatter_(
                                    1, indices, 1.) * normal

                                mask = torch.zeros(len(reduce), len(
                                    self.primitives)).cuda()
                                reduce.data = mask.scatter_(
                                    1, indices, 1.) * reduce

        else:
            for train_step in range(train_steps):
                warm_up = (
                    epoch < self.warm_up_epochs
                )  # if epoch < warm_up_epochs, do warm up

                # tracked based on the global steps not the stage training
                # steps
                if (
                    train_step >= arch_adap_steps
                ):  # no architecture adap after arch_adap_steps steps
                    warm_up = 1

                if w_task_lr_scheduler is not None:
                    w_task_lr_scheduler.step()

                if a_task_lr_scheduler is not None:
                    a_task_lr_scheduler.step()

                train(
                    task,
                    self.model,
                    self.architect,
                    self.w_optim,
                    self.a_optim,
                    lr,
                    global_progress,
                    self.config,
                    warm_up,
                )

                if (
                    model_has_normalizer
                    and train_step < (arch_adap_steps - 1)
                    and not warm_up
                ):  # todo check if not warm_up is correct
                    self.model.normalizer["params"]["curr_step"] += 1
                    self.architect.v_net.normalizer["params"]["curr_step"] += 1

        w_task = OrderedDict(
            {
                layer_name: copy.deepcopy(layer_weight)
                for layer_name, layer_weight in self.model.named_weights()
                if layer_weight.grad is not None
            }
        )

        a_task = OrderedDict(
            {
                layer_name: copy.deepcopy(layer_alpha)
                for layer_name, layer_alpha in self.model.named_alphas()
                if layer_alpha.grad is not None
            }
        )

        # Log genotype
        # TODO: Solve the parsing with or without switches
        # TODO: Do we reduce skip_connections, paper notes only during
        # evaluation.
        genotype = self.model.genotype()

        if log_alphas:
            alpha_logger["normal_relaxed"].append(
                copy.deepcopy(self.model.alpha_normal)
            )
            alpha_logger["reduced_relaxed"].append(
                copy.deepcopy(self.model.alpha_reduce)
            )
            alpha_logger["all_alphas"].append(a_task)
            alpha_logger["normal_hierarchical"].append(
                copy.deepcopy(self.model.alpha_in_normal)
            )
            alpha_logger["reduced_hierarchical"].append(
                copy.deepcopy(self.model.alpha_in_reduce)
            )
            alpha_logger["normal_pairwise"].append(
                copy.deepcopy(self.model.alpha_pw_normal)
            )
            alpha_logger["reduced_pairwise"].append(
                copy.deepcopy(self.model.alpha_pw_reduce)
            )

        # for test data evaluation, turn off drop path
        if self.config.drop_path_prob > 0.0:
            self.model.drop_path_prob(0.0)

        # Also, remove skip-connection dropouts during evaluation,
        # evaluation is on the train-test set.
        self.model.drop_out_skip_connections(0.0)

        # Evaluate the model on the training-test tasks,
        # Limit the number of skip-connections
        with torch.no_grad():
            for batch_idx, batch in enumerate(task.test_loader):

                x_test, y_test = batch
                x_test = x_test.to(self.config.device, non_blocking=True)
                y_test = y_test.to(self.config.device, non_blocking=True)

                if num_of_skip_connections is not None \
                        and self.config.use_search_space_regularization:
                    gt.limit_skip_connections_alphas(
                        self.model.alpha_normal,
                        self.primitives, k=2,
                        nodes=self.config.nodes,
                        num_of_skip_connections=num_of_skip_connections)

                if isinstance(self.model, SearchCNNController):
                    logits = self.model(
                        x_test, sparsify_input_alphas=sparsify_input_alphas
                    )
                else:
                    logits = self.model(x_test)
                loss = self.model.criterion(logits, y_test)

                y_test_pred = logits.softmax(dim=1)

                prec1, prec5 = utils.accuracy(logits, y_test, topk=(1, 5))
                losses_logger.update(loss.item(), 1)
                top1_logger.update(prec1.item(), 1)

        # return dict(genotype=genotype, top1=top1)
        task_info = namedtuple(
            "task_info",
            [
                "genotype",
                "top1",
                "w_task",
                "a_task",
                "loss",
                "y_test_pred",
                "sparse_num_params",
            ],
        )
        task_info.w_task = w_task
        task_info.a_task = a_task
        task_info.loss = loss
        y_test_pred = y_test_pred
        task_info.y_test_pred = y_test_pred
        task_info.genotype = genotype
        # task_info.top1 = top1

        task_info.sparse_num_params = self.model.get_sparse_num_params(
            self.model.alpha_prune_threshold
        )

        return task_info


def train(
    task,
    model,
    architect,
    w_optim,
    alpha_optim,
    lr,
    global_progress,
    config,
    warm_up=False,
):

    model.train()

    for step, ((train_X, train_y), (val_X, val_y)) in enumerate(
        zip(task.train_loader, task.valid_loader)
    ):
        train_X, train_y = train_X.to(config.device), train_y.to(config.device)
        val_X, val_y = val_X.to(config.device), val_y.to(config.device)
        N = train_X.size(0)

        # phase 2. architect step (alpha)
        if not warm_up:  # only update alphas outside warm up phase
            alpha_optim.zero_grad()
            if config.do_unrolled_architecture_steps:
                architect.virtual_step(
                    train_X, train_y, lr, w_optim)  # (calc w`)
            architect.backward(train_X, train_y, val_X, val_y, lr, w_optim)

            alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(train_X)

        loss = model.criterion(logits, train_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()


class Architect:
    """ Compute gradients of alphas """

    def __init__(self, net, w_momentum, w_weight_decay, use_first_order_darts):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.use_first_order_darts = use_first_order_darts

    def set_network(self, net):
        self.net = net
        self.v_net = copy.deepcopy(net)

    def virtual_step(self, train_X, train_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(train_X, train_y)  # L_train(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network
            # weight have to be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(),
                                gradients):
                m = w_optim.state[w].get(
                    "momentum_buffer", 0.0) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay * w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def backward(self, train_X, train_y, val_X, val_y, xi, w_optim):
        """Compute loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y)  # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[: len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        # use first order approximation for darts
        if self.use_first_order_darts:
            with torch.no_grad():
                for alpha, da in zip(self.net.alphas(), dalpha):
                    alpha.grad = da

        else:  # 2nd order DARTS

            hessian = self.compute_hessian(dw, train_X, train_y)

            # update final gradient = dalpha - xi*hessian
            with torch.no_grad():
                for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                    alpha.grad = da - xi * h

    def compute_hessian(self, dw, train_X, train_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_train(w+, alpha) } - dalpha{ L_train(w-, alpha)})
                    / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        # dalpha { L_train(w+) }
        loss = self.net.loss(train_X, train_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas())

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2.0 * eps * d

        # dalpha { L_train(w-) }
        loss = self.net.loss(train_X, train_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas())

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p - n) / 2.0 * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
