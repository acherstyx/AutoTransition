import os
import random
import sys
import argparse

import yaml
from yacs.config import CfgNode as CN
import torch

from config.custom_config import add_custom_config

_C = CN()

# --------------------
#       system
# --------------------
_C.SYS = CN()
# pytorch multiprocess config
_C.SYS.MULTIPROCESS = True
_C.SYS.INIT_METHOD = 'tcp://localhost:21222'
# gpu devices on a single machine
_C.SYS.NUM_GPU = torch.cuda.device_count()
_C.SYS.GPU_DEVICES = list(range(torch.cuda.device_count()))
_C.SYS.NUM_SHARDS = 1
_C.SYS.SHARD_ID = 0
_C.SYS.DETERMINISTIC = False
_C.SYS.SEED = 222

# --------------------
# log & checkpoint
# --------------------
_C.LOG = CN()

_C.LOG.DIR = "./log"
_C.LOG.CHECKPOINT_SUBDIR = "checkpoint"  # the checkpoint is saved to `cfg.LOG.DIR/cfg.CHECKPOINT_SUBDIR`
_C.LOG.TENSORBOARD_SUBDIR = "tensorboard"  # similar to CHECKPOINT_SUBDIR

_C.LOG.LOG_FILE = "logging.log"  # file name, os.path.join(cfg.LOG.DIR, cfg.LOG.LOG_FILE)
_C.LOG.LOG_CONSOLE_LEVEL = "warning"

# --------------------
# dataset config
# --------------------
# config for each dataset
_C.DATASET = CN()
_C.DATASET.NAME = None

# --------------------
# dataloader config
# --------------------
_C.LOADER = CN()

_C.LOADER.COLLATE_FN = None
_C.LOADER.BATCH_SIZE = 1
_C.LOADER.SHUFFLE = True

_C.LOADER.PREFETCH = 2
_C.LOADER.NUM_WORKERS = os.cpu_count() // torch.cuda.device_count() \
    if _C.SYS.MULTIPROCESS and torch.cuda.device_count() else os.cpu_count()
_C.LOADER.MULTIPROCESSING_CONTEXT = None

# --------------------
# model config
# --------------------
_C.MODEL = CN()

_C.MODEL.NAME = None
# the effect rely on implement of the model
_C.MODEL.USE_CHECKPOINT = False

# enable this option will significantly slow down the training speed
_C.MODEL.DDP_FIND_UNUSED_PARAMETERS = False

# --------------------
# backbone config
# --------------------
_C.BACKBONE = CN()

# --------------------
# loss config
# --------------------
_C.LOSS = CN()

# --------------------
# meter config
# --------------------
_C.METER = CN()

# --------------------
# train config
# --------------------
_C.TRAIN = CN()
_C.TRAIN.ENABLE = True
_C.TRAIN.EPOCH = 100
_C.TRAIN.EVAL_PERIOD = 1
_C.TRAIN.SAVE_PERIOD = 1
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.RESUME = None
_C.TRAIN.REWRITE_RESUME_MODEL = None
# apex amp
_C.TRAIN.AMP_ENABLE = False
# optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adam"
_C.TRAIN.OPTIMIZER.INIT_LR = 1e-3
# warmup scheduler
_C.TRAIN.SCHEDULER_WARMUP = CN()
_C.TRAIN.SCHEDULER_WARMUP.ENABLE = True
_C.TRAIN.SCHEDULER_WARMUP.METHOD = "exponential"
_C.TRAIN.SCHEDULER_WARMUP.EPOCH = 20
_C.TRAIN.SCHEDULER_WARMUP.INIT_LR = 1e-6
# scheduler
_C.TRAIN.SCHEDULER = CN()
_C.TRAIN.SCHEDULER.METHOD = "step"
# config for different type of schedulers
_C.TRAIN.SCHEDULER.STEP = CN()
_C.TRAIN.SCHEDULER.STEP.LR_DECAY_EPOCH = 30
_C.TRAIN.SCHEDULER.STEP.LR_DECAY_GAMMA = 0.1
_C.TRAIN.SCHEDULER.EXPONENTIAL = CN()
_C.TRAIN.SCHEDULER.EXPONENTIAL.GAMMA = 0.9
# train loss
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = None
_C.TRAIN.LOSS.CLIP_NORM = 0.1
_C.TRAIN.LOSS.WEIGHT = []
# train meter
_C.TRAIN.METER = None

# --------------------
# test config
# --------------------

_C.TEST = CN()
_C.TEST.ENABLE = False
# resume options
_C.TEST.AUTO_RESUME = True
_C.TEST.RESUME = None
# test meter
_C.TEST.METER = None

# ------------------------------------------------------------
# functions
# ------------------------------------------------------------

add_custom_config(_C)
default_config = _C.clone()
default_config.freeze()


def parse_args():
    parser = argparse.ArgumentParser(description="Special Effect Experimental Environment")
    parser.add_argument("--cfg", "-c",
                        help="path to the additional config file",
                        default=None,
                        type=str)
    parser.add_argument("opts",
                        help="see config/config.py for all options",
                        default=None,
                        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        print("Use default config...")
    return parser.parse_args()


def check_config(cfg: CN):
    # define some check
    assert not cfg.SYS.MULTIPROCESS or cfg.SYS.NUM_GPU > 0, "NUM_GPU should greater than 0 if MULTIPROCESS is enabled"
    assert cfg.SYS.NUM_GPU == len(cfg.SYS.GPU_DEVICES), "check GPU config"
    assert cfg.SYS.SHARD_ID < cfg.SYS.NUM_SHARDS


def get_config(args):
    config = _C.clone()
    if args.cfg is not None:
        config.merge_from_file(args.cfg)
    if args.opts is not None:
        config.merge_from_list(args.opts)
    config.freeze()
    check_config(config)
    return config


def dump():
    args = parse_args()
    cfg = get_config(args)
    with open("current_config.yaml", "w") as f:
        f.write(cfg.dump())


def summary(cfg: _C):
    print(f"Model: {cfg.MODEL.NAME}")
    print(f"Dataset: {cfg.DATASET.NAME}")


if __name__ == '__main__':
    dump()
