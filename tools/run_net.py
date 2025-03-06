# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 12:31
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : run_net.py

import logging
import os

import torch.multiprocessing as mp

from autotransition.config import parse_args, get_config
from autotransition.tools.test_net import test
from autotransition.tools.train_net import train
from autotransition.utils import (
    get_timestamp,
    setup_logging,
    init_distributed,
    save_config,
    manual_seed
)

logger = logging.getLogger(__name__)


# noinspection DuplicatedCode
def main():
    args = parse_args()
    # get config
    cfg = get_config(args)

    if cfg.TRAIN.ENABLE:
        print(f"{get_timestamp()} => Training")
        if cfg.SYS.MULTIPROCESS:
            mp.spawn(run_worker, args=(cfg, train), nprocs=cfg.SYS.NUM_GPU)
        else:
            run_worker(proc=0, cfg=cfg, func=train)
    else:
        print(f"{get_timestamp()} => Training is disabled")

    if cfg.TEST.ENABLE:
        print(f"{get_timestamp()} => Testing")
        if cfg.SYS.MULTIPROCESS:
            mp.spawn(run_worker, args=(cfg, test), nprocs=cfg.SYS.NUM_GPU)
        else:
            run_worker(proc=0, cfg=cfg, func=test)
    else:
        print(f"{get_timestamp()} => Testing is disabled")

    print(f"{get_timestamp()} => Finished!")


def run_worker(proc, cfg, func):
    setup_logging(cfg)

    init_distributed(proc, cfg)

    save_config(cfg)

    manual_seed(cfg)

    func(cfg)


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    main()
