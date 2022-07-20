# -*- coding: utf-8 -*-
# @Time    : 2021/12/25 11:31
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : build.py

import os
import logging

import torch.nn as nn
import torch.distributed as dist
from yacs.config import CfgNode

from fvcore.common.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

logger = logging.getLogger(__name__)


def build_model(cfg: CfgNode):
    model = MODEL_REGISTRY.get(cfg.MODEL.NAME)(cfg)

    # data parallel and distributed data parallel
    if cfg.SYS.MULTIPROCESS:
        logger.debug("Moving model to device: %s...", cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU])
        # move to GPU
        model.to(cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU])
        logger.debug("Model is moved to device: %s", cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU])
        return nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU]],
            output_device=cfg.SYS.GPU_DEVICES[dist.get_rank() % cfg.SYS.NUM_GPU],
            find_unused_parameters=cfg.MODEL.DDP_FIND_UNUSED_PARAMETERS
        )
    elif cfg.SYS.NUM_GPU > 0:
        model = model.cuda()
        return nn.parallel.DataParallel(model)
    else:
        return model
