###############################################################################
# Copyright (c) Hayeon Lee, Eunyoung Hyung [GitHub MetaD2A], 2021
# Rapid Neural Architecture Search by Learning to Generate Graphs from Datasets, ICLR 2021
###############################################################################
import os
import time
import tqdm
import numpy as np
from scipy.stats import pearsonr

import torch
from torch import optim

from metanas.meta_predictor.nas_bench_201 import train_single_model
from metanas.meta_predictor.predictor.model import PredictorModel
from metanas.meta_predictor.utils import (load_graph_config,
                                          load_pretrained_model,
                                          save_model, mean_confidence_interval)

from metanas.meta_predictor.loader import (get_meta_train_loader,
                                           MetaTestDataset)


class meta_predictor:
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.batch_size = config.pred_batch_size
        self.num_sample = config.num_samples
        self.epochs = config.pred_epochs

        self.save_epoch = config.save_epoch
        self.model_path = config.model_path
        self.save_path = config.save_path
        self.model_name = config.model_name
        self.data_path = config.data_path
        self.logger = config.logger
        self.test = config.test

        self.max_corr_dict = {'corr': -1, 'epoch': -1}
        self.train_arch = config.train_arch

        # NAS_bench 201 graph configuration
        graph_config = load_graph_config(
            config.graph_data_name, config.nvt, config.data_path)

        # Load predictor model
        self.model = PredictorModel(config, graph_config).to(self.device)
        if config.model_path is not None:
            load_pretrained_model(self.model_path, self.model)

        # Test when used as discrete estimate on the RL environment?
        if self.test:
            self.data_name = config.data_name
            self.num_class = config.num_class
            self.load_epoch = config.load_epoch
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min',
                factor=0.1,
                patience=10,
                verbose=True)

            self.mtrloader = get_meta_train_loader(
                self.batch_size, self.data_path, self.num_sample, is_pred=True)
            self.acc_mean = self.mtrloader.dataset.mean
            self.acc_std = self.mtrloader.dataset.std

    def forward(self, dataset, arch):
        D_mu = self.model.set_encode(dataset.to(self.device))
        G_mu = self.model.graph_encode(arch)

        y_pred = self.model.predict(D_mu, G_mu)
        return y_pred

    def meta_train(self):
        for epoch in range(1, self.epochs+1):
            loss, corr = self.meta_train_epoch(epoch)
            self.scheduler.step(loss)

            # self.config.logger print pred log train
            self.logger.info(
                f"Train epoch: {epoch:3d} "
                f"meta-training loss: {loss:.6f} "
                f"meta-training correlation: {corr:.4f}"
            )

            valoss, vacorr = self.meta_validation(epoch)
            if self.max_corr_dict['corr'] < vacorr:
                self.max_corr_dict['corr'] = vacorr
                self.max_corr_dict['epoch'] = epoch
                self.max_corr_dict['loss'] = valoss
                save_model(self.model_path, self.model, epoch, max_corr=True)

            # self.config.logger print pred log validation
            max_loss = self.max_corr_dict['loss']
            max_corr = self.max_corr_dict['corr']
            self.logger.info(
                f"Train epoch: {epoch:3d} "
                f"validation loss: {loss:.6f} ({max_loss:.6f})"
                f"max correlation correlation: {corr:.4f} ({max_corr:.4f})"
            )

            if epoch % self.save_epoch == 0:
                save_model(self.model_path, self.model, epoch)

    def meta_train_epoch(self):
        self.model.train()
        self.mtrloader.dataset.set_mode('train')
        N = len(self.mtrloader.dataset)

        trloss = 0
        y_all, y_pred_all = [], []

        for x, g, acc in tqdm(self.mtrloader):
            self.optimizer.zero_grad()
            y_pred = self.forward(x, g)
            y = acc.to(self.device)
            loss = self.model.mseloss(y_pred, y.unsqueeze(-1))
            loss.backward()
            self.optimizer.step()

            y = y.tolist()
            y_pred = y_pred.squeeze().tolist()
            y_all += y
            y_pred_all += y_pred
            trloss += float(loss)

        return trloss/N, pearsonr(np.array(y_all),
                                  np.array(y_pred_all))[0]

    def meta_validation(self):
        self.model.eval()
        self.mtrloader.dataset.set_mode('valid')

        valoss = 0
        N = len(self.mtrloader.dataset)
        y_all, y_pred_all = [], []

        with torch.no_grad():
            for x, g, acc in tqdm(self.mtrloader):
                y_pred = self.forward(x, g)
                y = acc.to(self.device)
                loss = self.model.mseloss(y_pred, y.unsqueeze(-1))

                y = y.tolist()
                y_pred = y_pred.squeeze().tolist()
                y_all += y
                y_pred_all += y_pred
                valoss += float(loss)

        return valoss/N, pearsonr(np.array(y_all),
                                  np.array(y_pred_all))[0]

    def evaluate_architecture(self, dataset):
        # For reward calculation
        # for data_name in ['cifar10', 'cifar100', 'mnist',
        # 'svhn', 'aircraft', 'pets']:

        # TODO: Refactor path
        self.nasbench201 = torch.load(
            os.path.join(self.data_path, 'nasbench201.pt'))

        self.test_dataset = MetaTestDataset(
            self.data_path, dataset, self.num_sample, self.num_class)

        # TODO: neural_network to graph

        # # gen_arch_str = self.load_generated_archs(data_name, run)
        # # gen_arch_igraph = self.get_items(
        # #     full_target=self.nasbench201['arch']['igraph'],
        # #     full_source=self.nasbench201['arch']['str'],
        # #     source=gen_arch_str)
        # y_pred_all = []

        self.model.eval()

        # sttime = time.time()
        # with torch.no_grad():
        #     for i, arch_igraph in enumerate(gen_arch_igraph):
        #         x, g = self.collect_data(arch_igraph)
        #         y_pred = self.forward(x, g)
        #         y_pred = torch.mean(y_pred)
        #         y_pred_all.append(y_pred.cpu().detach().item())

        # top_arch_lst = self.select_top_arch(
        #     data_name, torch.tensor(y_pred_all), gen_arch_str, N)
        # arch_runs.append(top_arch_lst[0])
        # elasped = time.time() - sttime
        # elasped_time.append(elasped)

    def meta_test(self, data_name):
        # TODO: Refactor this
        self.nasbench201 = torch.load(
            os.path.join(self.data_path, 'nasbench201.pt'))
        self.test_dataset = MetaTestDataset(
            self.data_path, data_name, self.num_sample, self.num_class)

        meta_test_path = os.path.join(
            self.save_path, 'meta_test', data_name, 'best_arch')
        if not os.path.exists(meta_test_path):
            os.makedirs(meta_test_path)
        f_arch_str = open(
            os.path.join(meta_test_path, 'architecture.txt'), 'w')
        save_path = os.path.join(meta_test_path, 'accuracy.txt')
        f = open(save_path, 'w')
        arch_runs = []
        elasped_time = []

        if 'cifar' in data_name:
            N = 30
            runs = 10
            acc_runs = []
        else:
            N = 1
            runs = 1

        print(
            f'==> select top architectures for {data_name} by meta-predictor...')
        for run in range(1, runs + 1):
            print(f'==> run #{run}')
            gen_arch_str = self.load_generated_archs(data_name, run)
            gen_arch_igraph = self.get_items(
                full_target=self.nasbench201['arch']['igraph'],
                full_source=self.nasbench201['arch']['str'],
                source=gen_arch_str)
            y_pred_all = []
            self.model.eval()

            sttime = time.time()
            with torch.no_grad():
                for i, arch_igraph in enumerate(gen_arch_igraph):
                    x, g = self.collect_data(arch_igraph)
                    y_pred = self.forward(x, g)
                    y_pred = torch.mean(y_pred)
                    y_pred_all.append(y_pred.cpu().detach().item())

            top_arch_lst = self.select_top_arch(
                data_name, torch.tensor(y_pred_all), gen_arch_str, N)
            arch_runs.append(top_arch_lst[0])
            elasped = time.time() - sttime
            elasped_time.append(elasped)

            if 'cifar' in data_name:
                acc = self.select_top_acc(data_name, top_arch_lst)
                acc_runs.append(acc)

        for run, arch_str in enumerate(arch_runs):
            f_arch_str.write(f'{arch_str}\n')
            print(f'{arch_str}')

        time_path = os.path.join(
            self.save_path, 'meta_test', data_name, 'time.txt')
        with open(time_path, 'a') as f_time:
            msg = f'predictor average elasped time {np.mean(elasped_time):.2f}s'
            print(f'==> save time in {time_path}')
            f_time.write(msg+'\n')
            print(msg)

        if self.train_arch:
            if not 'cifar' in data_name:
                acc_runs = self.train_single_arch(
                    data_name, arch_runs[0], meta_test_path)
            print(f'==> save results in {save_path}')
            for r, acc in enumerate(acc_runs):
                msg = f'run {r+1} {acc:.2f} (%)'
                f.write(msg+'\n')
                print(msg)

            m, h = mean_confidence_interval(acc_runs)
            msg = f'Avg {m:.2f}+-{h.item():.2f} (%)'
            f.write(msg+'\n')
            print(msg)

    """Meta-testing functions"""

    def train_single_arch(self, data_name, arch_str, meta_test_path):
        seeds = (777, 888, 999)
        train_single_model(save_dir=meta_test_path,
                           workers=8,
                           datasets=[data_name],
                           xpaths=[f'{self.data_path}/raw-data/{data_name}'],
                           splits=[0],
                           use_less=False,
                           seeds=seeds,
                           model_str=arch_str,
                           arch_config={'channel': 16, 'num_cells': 5})
        epoch = 49 if data_name == 'mnist' else 199
        test_acc_lst = []
        for seed in seeds:
            result = torch.load(os.path.join(
                meta_test_path, f'seed-0{seed}.pth'))
            test_acc_lst.append(
                result[data_name]['valid_acc1es'][f'x-test@{epoch}'])
        return test_acc_lst

    def select_top_arch_acc(
            self, data_name, y_pred_all, gen_arch_str, N):
        _, sorted_idx = torch.sort(y_pred_all, descending=True)
        gen_test_acc = self.get_items(
            full_target=self.nasbench201['test-acc'][data_name],
            full_source=self.nasbench201['arch']['str'],
            source=gen_arch_str)
        sorted_gen_test_acc = torch.tensor(gen_test_acc)[sorted_idx]
        sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]

        max_idx = torch.argmax(sorted_gen_test_acc[:N]).item()
        final_acc = sorted_gen_test_acc[:N][max_idx]
        final_str = sotred_gen_arch_str[:N][max_idx]
        return final_acc, final_str

    def select_top_arch(
            self, data_name, y_pred_all, gen_arch_str, N):
        _, sorted_idx = torch.sort(y_pred_all, descending=True)
        sotred_gen_arch_str = [gen_arch_str[_] for _ in sorted_idx]
        final_str = sotred_gen_arch_str[:N]
        return final_str

    def select_top_acc(self, data_name, final_str):
        final_test_acc = self.get_items(
            full_target=self.nasbench201['test-acc'][data_name],
            full_source=self.nasbench201['arch']['str'],
            source=final_str)
        max_test_acc = max(final_test_acc)
        return max_test_acc

    def collect_data(self, arch_igraph):
        x_batch, g_batch = [], []
        for _ in range(10):
            x_batch.append(self.test_dataset[0])
            g_batch.append(arch_igraph)
        return torch.stack(x_batch).to(self.device), g_batch

    def get_items(self, full_target, full_source, source):
        return [full_target[full_source.index(_)] for _ in source]

    def load_generated_archs(self, data_name, run):
        mtest_path = os.path.join(
            self.save_path, 'meta_test', data_name, 'generated_arch')
        with open(os.path.join(mtest_path, f'run_{run}.txt'), 'r') as f:
            gen_arch_str = [_.split()[0] for _ in f.readlines()[1:]]
        return gen_arch_str
