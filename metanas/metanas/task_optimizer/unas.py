import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import OrderedDict, namedtuple

from metanas.utils import utils
from metanas.models.search_cnn import SearchCNNController

import matplotlib.pyplot as plt

from metanas.utils import genotypes as gt
from metanas.task_optimizer import Alphas

class UNAS:
    def __init__(self, model, config, do_scheduler_lr=False):
        self.config = config
        self.model = model
        self.do_schedule_lr = do_scheduler_lr

        self.alphas = Alpha(num_normal=1,
                  num_reduce=1,
                  num_op=len(gt.PRIMITIVES_FEWSHOT), # TODO: Should be changing with P-DARTS settings
                  num_nodes=config.n_nodes,
                  gsm_soften_eps=0.0, # TODO: Add to config
                  gsm_temperature=0.4, # TODO: Add to config
                  same_alpha_minibatch=args.same_alpha_minibatch)

        self.task_train_steps = config.task_train_steps
        self.test_task_train_steps = config.test_task_train_steps
        self.warm_up_epochs = config.warm_up_epochs

        # weights optimizer
        self.w_optim = torch.optim.Adam(
            self.model.weights(),
            lr=self.config.w_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.w_weight_decay,
        ) 

        # architecture optimizer, self.arch_topimizer
        self.a_optim = torch.optim.Adam(
            model.alphas(),
            self.config.alpha_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.alpha_weight_decay,
        )


        self.architect = REINFORCE(
            self.model,
            self.config.w_momentum,
            self.config.w_weight_decay
        )

        # weights generalization error
        self.gen_error_alpha = 0.1 #args.gen_error_alpha
        self.gen_error_alpha_lambda = 0.1 #args.gen_error_alpha_lambda

        self.loss = None
        self.accuracy = None
        self.count = None
        self.loss_diff_sign = None
        self.reset_counter()

    def reset_counter(self):
        """Resets counters."""
        self.count = 0
        self.loss = utils.AverageMeter()
        self.accuracy = utils.AverageMeter()
        self.loss_diff_sign = utils.AverageMeter()

    def mean_accuracy(self):
        """Return mean accuracy."""
        return self.accuracy.avg

    def step(
        self,
        task,
        epoch,
        global_progress="",
        test_phase=False,
        alpha_logger=None,
        sparsify_input_alphas=None,
        switches_normal=None,  # P-DARTS variables
        switches_reduce=None,
        num_of_sk=None,
        dropout_sk=0.0
    ):
        # Configure variables
        log_alphas = False

        if test_phase:
            top1_logger = self.config.top1_logger_test
            losses_logger = self.config.losses_logger_test
            train_steps = self.config.test_task_train_steps
            arch_adap_steps = int(train_steps * self.config.test_adapt_steps)

            if alpha_logger is not None:
                log_alphas = True

        else:
            top1_logger = self.config.top1_logger
            losses_logger = self.config.losses_logger
            train_steps = self.config.task_train_steps
            arch_adap_steps = train_steps

        lr = self.config.w_lr

        for train_step in range(train_steps):

            warm_up = (
                epoch < self.warm_up_epochs
            )

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

        # TODO: here should be added logging and removal of droppath etc.

        #####

        # Test evaluation model at the end of training,
        with torch.no_grad():
            for batch_idx, batch in enumerate(task.test_loader):

                x_test, y_test = batch
                x_test = x_test.to(self.config.device, non_blocking=True)
                y_test = y_test.to(self.config.device, non_blocking=True)

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
        task_info.w_task = None  # w_task
        task_info.a_task = None  # a_task
        # task_info.loss = loss
        y_test_pred = y_test_pred
        task_info.y_test_pred = y_test_pred
        # task_info.genotype = genotype
        # # task_info.top1 = top1

        # task_info.sparse_num_params = self.model.get_sparse_num_params(
        #     self.model.alpha_prune_threshold
        # )
        raise task_info


def train(task,
          model,
          architect,
          w_optim,
          alpha_optim,
          lr,
          global_progress,
          config,
          warm_up=False):
    # Train loop of train_search.py in UNAS.
    model.train()

    for step, ((train_X, train_y), (val_X, val_y)) in enumerate(
        zip(task.train_loader, task.valid_loader)
    ):
        train_X, train_y = train_X.to(config.device), train_y.to(config.device)
        val_X, val_y = val_X.to(config.device), val_y.to(config.device)
        N = train_X.size(0)

        # TODO: Originally done in train step in UNAS
        # c_... variables are copies with no grad and clone of w_reduce
        # TODO: Refactor to alphas module?
        w_reduce, w_normal, c_reduce, c_normal = architect.sample_alphas()

        # phase 2. architect step (alpha)
        # TODO: Do we pass the validation data here or no?
        architect.step(train_X, train_y, val_X, val_y,
                       alpha_optim, w_optim,
                       w_reduce, w_normal,
                       global_progress)

        # phase 1. child network step (w)
        # TODO: Pass alphas, however still ignoring the pairwise alphas
        w_optim.zero_grad()
        logits = model(train_X, c_reduce, c_normal)
        loss = model.criterion(logits, train_y)

        # TODO: We don't have the grad_fn so this is required?
        # dummy = sum([torch.sum(param) for param in model.parameters()])
        # loss += dummy * 0.

        loss.backward()
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()


class REINFORCE:
    # This is the architect in the standard DARTS implementation
    # TODO: Could also attempt, REBAR and PPO?

    def __init__(self, model, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """

        self.model = model
        self.v_model = copy.deepcopy(model)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

        self.

        # Reset counters
        self.count = 0
        self.loss = utils.AverageMeter()
        self.loss_diff_sign = utils.AverageMeter()
        self.accuracy = utils.AverageMeter()

        self.exp_avg1 = utils.EMAMeter(alpha=0.9)
        self.exp_avg2 = utils.EMAMeter(alpha=0.9)

        self.gen_error_alpha = 0.2  # args.gen_error_alpha
        self.gen_error_alpha_lambda = 0.2  # args.gen_error_alpha_lambda

    def step(self, train_X, train_y, val_X, val_y, alpha_optim, w_optim,
             w_normal, w_reduce,
             global_progress):

        # TODO: find way of applying to pw alphas?
        # Discretize alphas

        alpha_optim.zero_grad()

        with torch.no_grad():
            #     # self.model.alphas
            #     # TODO: discritize alphas and adjust training obj
            #     # disc_weights = self.alpha.module.discretize(weights)
            d_normal = self.model.discretize_alphas(w_normal)
            d_reduce = self.model.discretize_alphas(w_reduce)
            loss_disc, acc, loss_train, loss_diff = self.training_objective(
                train_X, train_y,
                d_normal, d_reduce,
                val_X, val_y
            )

        # TODO: Possibly reduce_loss_disc, to one disk
        avg = torch.mean(loss_disc).detach()
        print("avg:", avg)

        # TODO: Write out formula
        # Baseline c_n?
        baseline = self.exp_avg1.avg
        self.exp_avg1.update(avg)  # Update moving average

        reward = (loss_disc - baseline).detach()
        # TODO: pull code from alphas module and more sure this
        # works for our implementation as well
        log_q_d = self.log_probability(w_normal, w_reduce, d_normal, d_reduce)
        loss = torch.mean(log_q_d * reward) + baseline
        print(log_q_d)
        print(reward)
        print(baseline)
        print(loss)

        loss_train, loss_diff = torch.mean(loss_train), torch.mean(loss_diff)

        # TODO: Log entropy possibly
        # entropy_loss = self.entropy_loss(w_normal, w_reduce)

        # Backward pass and update.
        # TODO: Create alpha module to be able to do backwards on the loss
        loss.backward()
        alpha_optim.step()

        self.loss.update(loss.data, 1)
        self.accuracy.update(acc, 1)
        self.count += 1

        # TODO: Possible reporting here

    def log_probability(self, w_normal, w_reduce, d_normal, d_reduce):
        def log_q(d, a):
            return torch.sum(torch.sum(d * a, dim=-1) - torch.logsumexp(
                a, dim=-1), dim=0)

        log_q_d = 0
        for n_edges, d_edges in zip(w_normal, d_normal):
            for i in range(len(n_edges)):
                a = n_edges[i]
                w = d_edges[i]
                log_q_d += log_q(w, a)

        for n_edges, d_edges in zip(w_reduce, d_reduce):
            for i in range(len(n_edges)):
                a = n_edges[i]
                w = d_edges[i]
                log_q_d += log_q(w, a)
        return log_q_d

    def entropy_loss(self, w_normal, w_reduce):
        def entropy(logit):
            q = F.softmax(logit, dim=-1)
            return - torch.sum(torch.sum(
                q * logit, dim=-1) - torch.logsumexp(logit, dim=-1), dim=0)

        entr = 0
        for n_edges in w_normal:
            for i in range(len(n_edges)):
                logit = n_edges[i]
                entr += entropy(logit)

        for n_edges in w_reduce:
            for i in range(len(n_edges)):
                logit = n_edges[i]
                entr += entropy(logit)
        return torch.mean(entr, dim=0)

    def sample_alphas(self):
        """Sample new weights,
        TODO: Make temperature configurable also do we anneal
        the temperature in sample lphas
        """
        sample_reduce, sample_normal = self.model.sample_alphas()

        # TODO: The original implementation also supplies a copy
        # of the alphas.
        c_reduce, c_normal = copy.deepcopy(
            sample_reduce), copy.deepcopy(sample_normal)
        return sample_reduce, sample_normal, c_reduce, c_normal

    def training_objective(self, X_train, y_train, w_normal, w_reduce,
                           X_val, y_val):
        # TODO: Pass more variables
        # def training_obj(self, train, train_target, weights,
        #                  model_opt, val, val_target, global_step):

        logits_train = self.model(X_train, w_normal, w_reduce)
        loss_train = self.model.loss(X_train, y_train)
        print("loss:", loss_train)

        # TODO: This is actually test?
        # logits_val = self.model(X_val)
        loss_val = self.model.loss(X_val, y_val)

        loss_diff = torch.abs(loss_val - loss_train)
        self.loss_diff_sign.update(
            torch.mean(((loss_val - loss_train) > 0).float()).data)

        # TODO: loss = loss + self.gen_error_alpha_lambda * loss_diff
        loss = loss_train + self.gen_error_alpha_lambda * loss_diff

        _, pred_train = torch.max(logits_train, dim=-1)
        accuracy = utils.accuracy(pred_train, y_train)

        return loss, accuracy, loss_train, loss_diff