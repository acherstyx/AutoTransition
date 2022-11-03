# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 13:31
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : build.py

import logging
from typing import *

from yacs.config import CfgNode
from fvcore.common.registry import Registry

import torch.distributed as dist
from torch.utils import data

DATASET_REGISTRY = Registry("DATASET")
DALI_PIPELINE_REGISTER = Registry("DALI_PIPELINE")
COLLATE_FN_REGISTER = Registry("COLLATE_FN")

logger = logging.getLogger(__name__)


def build_loader(cfg: CfgNode, mode=("train", "test", "val")):
    assert cfg.DATASET.NAME is not None, "specify a dataset to load in config: DATASET.NAME"

    if not isinstance(mode, tuple):
        mode = (mode,)

    dataset_builder: Callable[[CfgNode, str], data.Dataset] = DATASET_REGISTRY.get(cfg.DATASET.NAME)
    set_list = [dataset_builder(cfg, _mode) for _mode in mode]

    if cfg.SYS.MULTIPROCESS:
        sampler_list = [data.distributed.DistributedSampler(dataset,
                                                            rank=dist.get_rank(),
                                                            shuffle=cfg.LOADER.SHUFFLE)
                        for _mode, dataset in zip(mode, set_list)]
    else:
        sampler_list = [None for _mode in mode]
    collate_fn = COLLATE_FN_REGISTER.get(str(cfg.LOADER.COLLATE_FN)) if cfg.LOADER.COLLATE_FN is not None else None
    kwargs_default = {
        "batch_size": cfg.LOADER.BATCH_SIZE,
        "num_workers": cfg.LOADER.NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": bool(cfg.LOADER.NUM_WORKERS),
        "shuffle": False if cfg.SYS.MULTIPROCESS else cfg.LOADER.SHUFFLE,
        "prefetch_factor": cfg.LOADER.PREFETCH,
        "collate_fn": collate_fn,
        "multiprocessing_context": cfg.LOADER.MULTIPROCESSING_CONTEXT if cfg.SYS.MULTIPROCESS else None
    }
    loader_list = [data.DataLoader(dataset=_dataset, sampler=_sampler, **kwargs_default)
                   for _mode, _dataset, _sampler in zip(mode, set_list, sampler_list)]

    if cfg.SYS.MULTIPROCESS and all(sampler is not None for sampler in sampler_list):
        res = {}
        res.update({_mode: loader for _mode, loader in zip(mode, loader_list)})
        res.update({f"{_mode}_sampler": sampler for _mode, sampler in zip(mode, sampler_list)})
        return res
    else:
        return {_mode: loader for _mode, loader in zip(mode, loader_list)}


@COLLATE_FN_REGISTER.register()
def dummy_collate_fn(x):
    return x
