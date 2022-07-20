# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 11:45 AM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : build.py

import torch.nn as nn
from yacs.config import CfgNode
from fvcore.common.registry import Registry

BACKBONE_REGISTER = Registry("BACKBONE")


def build_backbone(cfg: CfgNode, backbone_name: str) -> nn.Module:
    """
    build backbone base on backbone config
    :param cfg: backbone config
    :param backbone_name:
    """
    backbone = BACKBONE_REGISTER.get(backbone_name)(cfg)

    return backbone
