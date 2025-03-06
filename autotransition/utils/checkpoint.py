# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 16:51
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : checkpoint.py

import logging
import os
from collections import OrderedDict
from typing import *

import torch

logger = logging.getLogger(__name__)


def auto_resume(ckpt_folder):
    try:
        ckpt_files = [ckpt for ckpt in os.listdir(ckpt_folder) if ckpt.endswith(".pth")]
    except FileNotFoundError:
        ckpt_files = []
    if len(ckpt_files) > 0:
        return max([os.path.join(ckpt_folder, file) for file in ckpt_files], key=os.path.getmtime)
    else:
        return None


def save_checkpoint(ckpt_folder, epoch, model, optimizer, scheduler, config, prefix="checkpoint"):
    stat_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else "",
        "config": config
    }
    ckpt_path = os.path.join(ckpt_folder, f"{prefix}_{epoch}.pth")
    os.makedirs(ckpt_folder, exist_ok=True)
    torch.save(stat_dict, ckpt_path)
    return ckpt_path


def load_checkpoint(ckpt_file, model: torch.nn.Module, optimizer: Union[torch.optim.Optimizer, None], scheduler: Any,
                    restart_train=False, rewrite: Tuple[str, str] = None):
    state_dict = torch.load(ckpt_file, map_location="cpu")
    if rewrite is not None:
        logger.info("rewrite model checkpoint prefix: %s->%s", *rewrite)
        state_dict["model"] = {k.replace(*rewrite) if k.startswith(rewrite[0]) else k: v
                               for k, v in state_dict["model"].items()}
    try:
        missing = model.load_state_dict(state_dict["model"], strict=False)
        logger.debug(f"checkpoint key missing: {missing}")
    except RuntimeError:
        print("fail to directly recover from checkpoint, try to match each layers...")
        net_dict = model.state_dict()
        print("find %s layers", len(state_dict["model"].items()))
        missing_keys = [k for k, v in state_dict["model"].items() if k not in net_dict or net_dict[k].shape != v.shape]
        print("missing key: %s", missing_keys)
        state_dict["model"] = {k: v for k, v in state_dict["model"].items() if
                               (k in net_dict and net_dict[k].shape == v.shape)}
        print("resume %s layers from checkpoint", len(state_dict["model"].items()))
        net_dict.update(state_dict["model"])
        model.load_state_dict(OrderedDict(net_dict))

    if not restart_train:
        if optimizer is not None and state_dict["optimizer"]:
            optimizer.load_state_dict(state_dict["optimizer"])
        if scheduler is not None and state_dict["scheduler"]:
            scheduler.load_state_dict(state_dict["scheduler"])
        epoch = state_dict["epoch"]
    else:
        logger.info("restart train, optimizer and scheduler will not be resumed")
        epoch = 0

    del state_dict
    torch.cuda.empty_cache()
    return epoch  # start epoch
