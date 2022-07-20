# -*- coding: utf-8 -*-
# @Time    : 2021/12/15 3:56 PM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : test_net.py
import logging
import os
from typing import *

import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

from model import build_model
from data import build_loader
from utils import build_meter, MeterBase, checkpoint, PreFetcher

logger = logging.getLogger(__name__)


@torch.no_grad()
def test(cfg: CfgNode):
    dataloader: Dict[str, Iterable] = build_loader(cfg, "test")
    model: nn.Module = build_model(cfg)
    model.eval()

    # resume from checkpoint: manual/auto
    is_auto_resumed = False
    epoch_start = None
    if cfg.TEST.AUTO_RESUME:
        ckpt_dir = os.path.join(cfg.LOG.DIR, cfg.LOG.CHECKPOINT_SUBDIR)
        logger.debug("auto resume search folder: %s", ckpt_dir)
        ckpt_file = checkpoint.auto_resume(ckpt_dir)
        if ckpt_file is not None:
            logger.info("auto resume from checkpoint file: %s", ckpt_file)
            if cfg.TEST.RESUME is not None:
                logger.warning("auto resume is enabled, specified checkpoint will be ignored: %s", cfg.TEST.RESUME)
            epoch_start = checkpoint.load_checkpoint(ckpt_file, model, None, None, restart_train=False)
            is_auto_resumed = True
        else:
            logger.info("auto resume is enabled, but no checkpoint is found in %s", ckpt_dir)
    if cfg.TEST.RESUME is not None and not is_auto_resumed:
        logger.info("resume from specified checkpoint: %s", cfg.TEST.RESUME)
        epoch_start = checkpoint.load_checkpoint(cfg.TEST.RESUME, model, None, None, restart_train=False)
    elif not is_auto_resumed:  # neither manual resume nor auto resume
        logger.warning("you are in test mode, but no pretrained model is loaded for testing!")

    writer = SummaryWriter(os.path.join(cfg.LOG.DIR, cfg.LOG.TENSORBOARD_SUBDIR), purge_step=0)

    test_meter: MeterBase = build_meter(cfg=cfg, name=cfg.TEST.METER, writer=writer, mode="test", purge_step=0)

    if torch.cuda.is_available() and cfg.SYS.NUM_GPU > 0:
        dataloader["test"] = PreFetcher(dataloader["test"])
    if not dist.is_initialized() or dist.get_rank() == 0:
        dataloader["test"] = tqdm.tqdm(dataloader["test"])
    for data, label in dataloader["test"]:
        output = model(data)
        test_meter.update(data, label, output)

    test_meter.summary(epoch=epoch_start)
    test_meter.reset()
