import copy
import torch
import torch.nn as nn
from collections import OrderedDict, namedtuple

from metanas.utils import utils


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
        )  #

        # architecture optimizer
        self.a_optim = torch.optim.Adam(
            model.alphas(),
            self.config.alpha_lr,
            betas=(0.0, 0.999),
            weight_decay=self.config.alpha_weight_decay,
        )

        # TODO: Architect from Darts will be Reinforce in this case
        # instead of bi-level optimization.
        self.architect = REINFORCE(
            self.model,
            self.config.w_momentum,
            self.config.w_weight_decay,
            self.config.use_first_order_darts,
        )

        # TODO: Define surrogate?
        self.loss = None
        self.accuracy = None
        self.count = None
        self.loss_diff_sign = None
        self.reset_counter()
        # TODO: Add?
        # self.report_freq = args.report_freq

    def reset_counter(self):
        """Resets counters."""
        self.count = 0
        self.loss = utils.AverageMeter()
        self.accuracy = utils.AverageMeter()
        self.loss_diff_sign = utils.AverageMeter()

    def mean_accuracy(self):
        """Return mean accuracy."""
        return self.accuracy.avg

    def training_obj(self, train, train_target, weights,
                     model_opt, val, val_target, global_step):

        # logits_train = self.model(train, weights)
        # loss_train = self.criterion(logits_train, train_target)

        # logits_val = self.model(val, weights)
        # loss_val = self.criterion(logits_val, val_target)

        # loss2 = torch.abs(loss_val - loss_train)
        # self.loss_diff_sign.update(torch.mean(
        #     ((loss_val - loss_train) > 0).float()).data)
        # loss1 = loss_train

        # loss = loss1 + self.gen_error_alpha_lambda * loss2
        # accuracy = utils.accuracy(logits_train, train_target)[0]

        # return loss, accuracy, loss1, loss2

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

        with torch.no_grad():
            # self.model.alphas
            self.training_objective()

        # TODO: Use train step
        # train(task, model, reinforce, w_opt, a_opt, lr, global_progress,
        # config, warm_up)
        # task train_steps = epochs per task
        for train_step in range(train_steps):

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
    raise NotImplementedError


class REINFORCE:
    # This is the architect in the standard DARTS implementation
    # TODO: Could also attempt, REBAR and PPO?

    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def training_objective(self):
        raise NotImplementedError
