# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 16:51
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : cache.py

import torch.distributed as dist

import logging
import os.path

import pickle

logger = logging.getLogger(__name__)


def cache_save(cache_file, metadata, match=None, barrier=True):
    state_dict = {"cache": metadata,
                  "match": match}
    if barrier and dist.is_initialized():
        dist.barrier()
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(cache_file, "wb") as f:
            pickle.dump(state_dict, f)


def try_cache_load(cache_file, match=None):
    if not os.path.exists(cache_file):
        logger.debug("cache file do not exist")
        return None

    try:
        with open(cache_file, "rb") as f:
            state_dict = pickle.load(f)
        if match == state_dict["match"]:
            logger.debug("cache load successfully.")
            return state_dict["cache"]
        else:
            logger.debug("cache mismatch.")
            return None
    except Exception as e:
        logger.warning("cache load failed because following error: %s", e)
        return None
