# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 16:11
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : train_utils.py

import datetime
import logging
import os.path
import random
import time
import typing

import numpy as np
import torch
import torch.distributed as dist
from fvcore.common.config import CfgNode

logger = logging.getLogger(__name__)


class Timer:
    def __init__(self):
        self.time = time.time()

    def __call__(self, reset=True):
        pre = self.time
        if reset:
            after = self.time = time.time()
        else:
            after = time.time()
        return (after - pre) * 1000


class PreFetcher:
    def __init__(self, data_loader):
        self.dl = data_loader
        self.loader = iter(data_loader)
        self.stream = torch.cuda.Stream()
        self.batch = None

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            self.batch = self.cuda(self.batch)

    @staticmethod
    def cuda(x: typing.Any):
        if isinstance(x, list) or isinstance(x, tuple):
            return [PreFetcher.cuda(i) for i in x]
        elif isinstance(x, dict):
            return {k: PreFetcher.cuda(v) for k, v in x.items()}
        elif isinstance(x, torch.Tensor):
            return x.cuda(non_blocking=True)
        else:
            return x

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def __iter__(self):
        self.preload()
        return self

    def __len__(self):
        return len(self.dl)


def manual_seed(cfg: CfgNode):
    if cfg.SYS.DETERMINISTIC:
        torch.manual_seed(cfg.SYS.SEED)
        random.seed(cfg.SYS.SEED)
        np.random.seed(cfg.SYS.SEED)
        torch.cuda.manual_seed(cfg.SYS.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        logger.debug("Manual seed is set")
    else:
        logger.warning("Manual seed is not used")


def init_distributed(proc: int, cfg: CfgNode):
    if cfg.SYS.MULTIPROCESS:  # initialize multiprocess
        word_size = cfg.SYS.NUM_GPU * cfg.SYS.NUM_SHARDS
        rank = cfg.SYS.NUM_GPU * cfg.SYS.SHARD_ID + proc
        logger.info("Initializing rank %s...", rank)
        dist.init_process_group(backend="nccl", init_method=cfg.SYS.INIT_METHOD, world_size=word_size, rank=rank)
        torch.cuda.set_device(cfg.SYS.GPU_DEVICES[proc])
        logger.info("Rank %s, shard %s initialized, GPU device is set to %s",
                    rank, cfg.SYS.SHARD_ID, cfg.SYS.GPU_DEVICES[proc])


def save_config(cfg: CfgNode):
    if not dist.is_initialized() or dist.get_rank() == 0:
        config_file = os.path.join(cfg.LOG.DIR, f"config-{get_timestamp()}.yaml")
        with open(config_file, "w") as f:
            f.write(cfg.dump())
        logger.debug("config is saved to %s", config_file)


def get_timestamp():
    """
    return a timestamp
    """
    return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
