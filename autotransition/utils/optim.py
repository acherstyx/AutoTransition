# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 16:12
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : optim.py

from torch import optim
from yacs.config import CfgNode


def build_optimizer(cfg: CfgNode, parameters) -> optim.Optimizer:
    if cfg.TRAIN.OPTIMIZER.NAME == "adam":
        optimizer = optim.Adam(params=parameters,
                               lr=cfg.TRAIN.SCHEDULER_WARMUP.INIT_LR if cfg.TRAIN.SCHEDULER_WARMUP.ENABLE
                               else cfg.TRAIN.OPTIMIZER.INIT_LR)
    elif cfg.TRAIN.OPTIMIZER.NAME == "sgd":
        optimizer = optim.SGD(params=parameters,
                              lr=cfg.TRAIN.SCHEDULER_WARMUP.INIT_LR if cfg.TRAIN.SCHEDULER_WARMUP.ENABLE
                              else cfg.TRAIN.OPTIMIZER.INIT_LR,
                              momentum=0.9)
    else:
        raise ValueError(f'optimizer "{cfg.TRAIN.OPTIMIZER.NAME}" is not defined')

    return optimizer


def build_scheduler(cfg: CfgNode,
                    optimizer: optim.Optimizer) -> super(optim.lr_scheduler.StepLR):  # typing: dummy type for return

    if cfg.TRAIN.SCHEDULER.METHOD == "step":
        return optim.lr_scheduler.StepLR(optimizer,
                                         step_size=cfg.TRAIN.SCHEDULER.STEP.LR_DECAY_EPOCH,
                                         gamma=cfg.TRAIN.SCHEDULER.STEP.LR_DECAY_GAMMA)
    if cfg.TRAIN.SCHEDULER.METHOD == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer,
                                                gamma=cfg.TRAIN.SCHEDULER.EXPONENTIAL.GAMMA)
    else:
        raise ValueError(f'scheduler "{cfg.TRAIN.SCHEDULER}" is not defined')


def build_warmup(cfg: CfgNode,
                 optimizer: optim.Optimizer):
    if cfg.TRAIN.SCHEDULER_WARMUP.METHOD == "exponential":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=(cfg.TRAIN.OPTIMIZER.INIT_LR / cfg.TRAIN.SCHEDULER_WARMUP.INIT_LR) **
                  (1 / cfg.TRAIN.SCHEDULER_WARMUP.EPOCH)
        )
    else:
        raise ValueError(f'warmup scheduler {cfg.TRAIN.SCHEDULER_WARMUP.METHOD} is not defined')
