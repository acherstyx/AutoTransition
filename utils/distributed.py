# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 1:23 PM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : distributed.py

from typing import *
import itertools

import torch
import torch.distributed as dist


def gather_object_multiple_gpu(list_object: List[Any]):
    """
    gather a list of something from multiple GPU
    :param list_object:
    """
    gathered_objects = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_objects, list_object)
    return list(itertools.chain(*gathered_objects))
