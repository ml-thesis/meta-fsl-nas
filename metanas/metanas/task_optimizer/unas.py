
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import copy
from collections import OrderedDict, namedtuple

from metanas.utils import utils
from metanas.utils import genotypes as gt
from metanas.models.search_cnn import SearchCNNController
from metanas.task_optimizer.alpha import Alpha

class UNAS:
    def __init__(self, model, config, do_scheduler_lr=False):
        self.config = config
        self.model = model
        self.do_schedule_lr = do_scheduler_lr

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

        # architecture optimizer
        self.a_optim = torch.optim.Adam(
            model.alphas(),
            self.config.alpha_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.alpha_weight_decay,
        )

        # TODO: For Factorized cell structure
        self.alpha = Alpha(self.model.alpha_normal, self.model.alpha_reduce)

        self.architect = REINFORCE(
            self.model,
            self.config,
            self.alpha,
            self.config.w_momentum,
            self.config.w_weight_decay
        )

        self.loss = None
        self.accuracy = None
        self.count = None
        self.loss_diff_sign = None
        self.reset_counter()

    def reset_counter(self):
        """Resets counters
        """
        self.count = 0
        self.loss = utils.AverageMeter()
        self.accuracy = utils.AverageMeter()
        self.loss_diff_sign = utils.AverageMeter()

    def mean_accuracy(self):
        """Return mean accuracy
        """
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
                self.alpha,
                lr,
                global_progress,
                self.config,
                warm_up,
            )

        # TODO: Add logging as done in darts.py

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
        return task_info


def train(task,
          model,
          architect,
          w_optim,
          alpha_optim,
          alpha,
          lr, # weight learning rate
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

        # w_reduce, w_normal, c_reduce, c_normal = architect.sample_alphas()
        w_normal, w_reduce = alpha()
        w_normal_no_grad, w_reduce_no_grad = alpha.clone_weights(w_normal, w_reduce)

        # phase 2. architect step (alpha)
        # TODO: Do we pass the validation data here or no?
        architect.step(train_X, train_y, val_X, val_y,
                       alpha_optim, w_optim,
                       w_reduce, w_normal,
                       global_progress)

        # phase 1. child network step (w)
        # TODO: Pass alphas, however still ignoring the pairwise alphas
        w_optim.zero_grad()

        model.set_alphas(w_normal_no_grad, w_reduce_no_grad)

        logits = model(train_X)
        loss = model.criterion(logits, train_y)

        # TODO: We don't have the grad_fn so this is required?
        # dummy = sum([torch.sum(param) for param in model.parameters()])
        # loss += dummy * 0.

        loss.backward()
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()


class REINFORCE:
    """This is the architect in the standard DARTS implementation
    """
    def __init__(self, model, config, alpha, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """

        self.model = model
        self.v_model = copy.deepcopy(model)

        self.alpha = alpha
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

        # Reset counters
        self.count = 0
        self.loss = utils.AverageMeter()
        self.loss_diff_sign = utils.AverageMeter()
        self.accuracy = utils.AverageMeter()

        self.exp_avg1 = utils.EMAMeter(alpha=0.9)
        self.exp_avg2 = utils.EMAMeter(alpha=0.9)

        self.gen_error_alpha = config.gen_error_alpha
        self.gen_error_alpha_lambda = config.gen_error_alpha_lambda 

    def step(self, train_X, train_y, val_X, val_y, alpha_optim, w_optim,
             w_normal, w_reduce,
             global_progress):

        # TODO: find way of applying to pw alphas?
        alpha_optim.zero_grad()

        with torch.no_grad():
            # TODO: Originally done by self.module.alphas
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
        log_q_d = self.alpha.log_probability(w_normal, w_reduce, d_normal, d_reduce)
        loss = torch.mean(log_q_d * reward) + baseline
        
        loss_train, loss_diff = torch.mean(loss_train), torch.mean(loss_diff)

        # TODO: Log entropy possibly
        # entropy_loss = self.entropy_loss(w_normal, w_reduce)

        # Backward pass and update.
        # TODO: Create alpha module to be able to do backwards on the loss
        loss.backward()
        alpha_optim.step()

        self.loss.update(loss.data, 1)
        # self.accuracy.update(acc, 1)
        self.count += 1

        # TODO: Possible reporting here

    def training_objective(self, X_train, y_train, w_normal, w_reduce,
                           X_val, y_val):

        if not self.gen_error_alpha:
            # TODO: Find way to pass alphas to model
            # model.set_alphas()
            self.model.set_alphas(w_normal, w_reduce)
            logits_train = self.model(X_train)
            loss_train = self.model.loss(X_train, y_train)

            _, pred_train = torch.max(logits_train, dim=-1)
            # accuracy = utils.accuracy(pred_train, y_train)

            # final loss
            loss_diff = 0
            loss = loss_train
        else:
            # TODO: Find way to pass alphas to model
            # model.set_alphas()
            self.model.set_alphas(w_normal, w_reduce)
            logits_train = self.model(X_train)
            loss_train = self.model.loss(X_train, y_train)

            # TODO: Double check
            loss_val = self.model.loss(X_val, y_val)
            loss_diff = torch.abs(loss_val - loss_train)

            self.loss_diff_sign.update(
                torch.mean(((loss_val - loss_train) > 0).float()).data)

            # TODO: loss = loss + self.gen_error_alpha_lambda * loss_diff
            loss = loss_train + self.gen_error_alpha_lambda * loss_diff

        accuracy = 0.5
        return loss, accuracy, loss_train, loss_diff