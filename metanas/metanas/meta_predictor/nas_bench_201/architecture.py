###############################################################
# NAS-Bench-201, ICLR 2020 (https://arxiv.org/abs/2001.00326) #
###############################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019.08           #
###############################################################

from nas_bench_201.log_utils import Logger, convert_secs2time
from nas_bench_201.models import CellStructure, CellArchitectures, get_search_spaces

from pathlib import Path
import time
import torch
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# from config_utils import load_config
# from procedures import save_checkpoint, copy_checkpoint
# from procedures import get_machine_info
# from functions import evaluate_for_seed


def train_single_model(save_dir, workers, datasets, xpaths, splits, use_less,
                       seeds, model_str, arch_config):
    assert torch.cuda.is_available(), 'CUDA is not available.'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(workers)

    save_dir = Path(save_dir)
    logger = Logger(str(save_dir), 0, False)

    if model_str in CellArchitectures:
        arch = CellArchitectures[model_str]
        logger.log(
            'The model string is found in pre-defined architecture dict : {:}'.format(model_str))
    else:
        try:
            arch = CellStructure.str2structure(model_str)
        except:
            raise ValueError(
                'Invalid model string : {:}. It can not be found or parsed.'.format(model_str))

    assert arch.check_valid_op(get_search_spaces(
        'cell', 'nas-bench-201')), '{:} has the invalid op.'.format(arch)
    # assert arch.check_valid_op(get_search_spaces('cell', 'full')), '{:} has the invalid op.'.format(arch)
    logger.log('Start train-evaluate {:}'.format(arch.tostr()))
    logger.log('arch_config : {:}'.format(arch_config))

    start_time, seed_time = time.time(), AverageMeter()
    for _is, seed in enumerate(seeds):
        logger.log(
            '\nThe {:02d}/{:02d}-th seed is {:} ----------------------<.>----------------------'.format(_is, len(seeds),
                                                                                                        seed))
        to_save_name = save_dir / 'seed-{:04d}.pth'.format(seed)
        if to_save_name.exists():
            logger.log(
                'Find the existing file {:}, directly load!'.format(to_save_name))
            checkpoint = torch.load(to_save_name)
        else:
            logger.log(
                'Does not find the existing file {:}, train and evaluate!'.format(to_save_name))
            checkpoint = evaluate_all_datasets(arch, datasets, xpaths, splits, use_less,
                                               seed, arch_config, workers, logger)
            torch.save(checkpoint, to_save_name)
        # log information
        logger.log('{:}'.format(checkpoint['info']))
        all_dataset_keys = checkpoint['all_dataset_keys']
        for dataset_key in all_dataset_keys:
            logger.log('\n{:} dataset : {:} {:}'.format(
                '-' * 15, dataset_key, '-' * 15))
            dataset_info = checkpoint[dataset_key]
            # logger.log('Network ==>\n{:}'.format( dataset_info['net_string'] ))
            logger.log('Flops = {:} MB, Params = {:} MB'.format(
                dataset_info['flop'], dataset_info['param']))
            logger.log('config : {:}'.format(dataset_info['config']))
            logger.log('Training State (finish) = {:}'.format(
                dataset_info['finish-train']))
            last_epoch = dataset_info['total_epoch'] - 1
            train_acc1es, train_acc5es = dataset_info['train_acc1es'], dataset_info['train_acc5es']
            valid_acc1es, valid_acc5es = dataset_info['valid_acc1es'], dataset_info['valid_acc5es']
        # measure elapsed time
        seed_time.update(time.time() - start_time)
        start_time = time.time()
        need_time = 'Time Left: {:}'.format(convert_secs2time(
            seed_time.avg * (len(seeds) - _is - 1), True))
        logger.log(
            '\n<<<***>>> The {:02d}/{:02d}-th seed is {:} <finish> other procedures need {:}'.format(_is, len(seeds), seed,
                                                                                                     need_time))
    logger.close()
