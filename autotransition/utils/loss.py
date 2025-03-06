# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 16:51
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : loss.py

import logging

import torch.nn as nn
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry

__all__ = ["LOSS_REGISTRY", "LossBase", "build_loss"]

LOSS_REGISTRY = Registry("LOSS")

logger = logging.getLogger(__name__)


class LossBase(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(LossBase, self).__init__()
        self.cfg = cfg

    def forward(self, outputs, labels):
        raise NotImplementedError


# register torch loss functions
for attr_name in nn.__dict__.keys():
    if "Loss" in attr_name:
        LOSS_REGISTRY.register(getattr(nn, attr_name))


def build_loss(cfg):
    loss_builder = LOSS_REGISTRY.get(cfg.TRAIN.LOSS.NAME)
    if issubclass(loss_builder, LossBase):
        return loss_builder(cfg)
    else:
        return loss_builder()
