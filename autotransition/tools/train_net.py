# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 12:13
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : train_net.py

import logging
import os
from typing import *

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from autotransition.data import build_loader
from autotransition.model import build_model
from autotransition.utils import build_meter, build_loss, load_checkpoint, save_checkpoint, auto_resume, \
    PreFetcher, Timer, build_scheduler, build_warmup, build_optimizer, MeterBase

logger = logging.getLogger(__name__)


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: optim.Optimizer,
                loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                meter: MeterBase,
                writer: torch.utils.tensorboard.SummaryWriter = None,
                clip_norm: int = None,
                epoch: int = 0):
    model.train()
    meter.set_mode("train")

    if torch.cuda.is_available():
        dataloader = PreFetcher(dataloader)  # prefetch to GPU
    dataloader = tqdm(dataloader,
                      desc=f"Train epoch {epoch + 1}") if not dist.is_initialized() or dist.get_rank() == 0 else dataloader
    timer = Timer()
    time_log = {}
    for cur_step, (inputs, labels) in enumerate(dataloader):
        time_log["load_data"] = timer()

        global_step = epoch * len(dataloader) + cur_step
        # forward
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        time_log["forward"] = timer()
        # backward
        optimizer.zero_grad()
        loss.backward()
        if clip_norm is not None:  # clip by norm
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        time_log["backward"] = timer()
        optimizer.step()
        time_log["optimize"] = timer()

        # summary
        with torch.no_grad():
            # gather loss, write loss to log
            if dist.is_initialized():
                dist.all_reduce(loss)
                loss = loss / dist.get_world_size()
                logger.debug(f"loss (rank {dist.get_rank()}, step {global_step}): {loss.cpu().detach().numpy()}")
            else:
                logger.debug(f"loss (no rank, step {global_step}): {loss.cpu().detach().numpy()}")

            if writer is not None:
                writer.add_scalars("train/timer", time_log, global_step=global_step)
                time_log = {}
                writer.add_scalar("train/loss", loss.detach(), global_step=global_step)
                writer.add_scalars("train/lr",
                                   {f"param_group_{i}_lr": group["lr"]
                                    for i, group in enumerate(optimizer.param_groups)},
                                   global_step=global_step)

            meter.update(inputs=inputs, labels=labels, outputs=outputs, n=None, global_step=global_step)
        time_log["summary"] = timer()
    meter.summary(epoch=epoch + 1)
    meter.reset()


@torch.no_grad()
def eval_epoch(model: nn.Module,
               dataloader: DataLoader,
               meter: MeterBase,
               writer: torch.utils.tensorboard.SummaryWriter = None,
               epoch: int = 0):
    model.eval()
    meter.set_mode("val")
    dataloader = tqdm(dataloader,
                      desc=f"Eval epoch {epoch + 1}") if not dist.is_initialized() or dist.get_rank() == 0 else dataloader
    if torch.cuda.is_available():
        dataloader = PreFetcher(dataloader)  # move to GPU
    for inputs, labels in dataloader:
        # forward
        outputs = model(inputs)

        with torch.no_grad():
            meter.update(inputs=inputs, labels=labels, outputs=outputs)
    meter.summary(epoch=epoch + 1)


def train(cfg):
    logger.debug("Building model...")
    model: nn.Module = build_model(cfg=cfg)  # build the model and move to GPU device properly
    logger.debug("Building dataloader...")
    dataloader: dict = build_loader(cfg=cfg, mode=("train", "val"))

    logger.debug("Building optimizer...")
    optimizer: optim.Optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg=cfg, optimizer=optimizer) if cfg.TRAIN.SCHEDULER.METHOD is not None else None
    warmup_scheduler = build_warmup(cfg=cfg, optimizer=optimizer) if cfg.TRAIN.SCHEDULER_WARMUP.ENABLE else None
    logger.debug("Done")

    # load checkpoint
    epoch_start = 0
    if cfg.TRAIN.AUTO_RESUME:
        logger.info("auto resume is enabled, recover from the most recent checkpoint first, "
                    "other checkpoints will be ignored")
        ckpt_dir = os.path.join(cfg.LOG.DIR, cfg.LOG.CHECKPOINT_SUBDIR)
        ckpt_file = auto_resume(ckpt_dir)
        if ckpt_file is not None:
            logger.info(f"auto resume from checkpoint: {ckpt_file}")
            if cfg.TRAIN.RESUME is not None:
                logger.warning(f"specified checkpoint {cfg.TRAIN.RESUME} will be ignored.")
            epoch_start = load_checkpoint(ckpt_file, model, optimizer, scheduler, restart_train=False)
        elif cfg.TRAIN.RESUME is not None:
            logger.info(f"resume from specified checkpoint {cfg.TRAIN.RESUME}.")
            epoch_start = load_checkpoint(cfg.TRAIN.RESUME, model, optimizer, scheduler, restart_train=True,
                                          rewrite=cfg.TRAIN.REWRITE_RESUME_MODEL)
        else:
            logger.info(f"no checkpoint was found in directory {ckpt_dir}")
    elif cfg.TRAIN.RESUME is not None:
        logger.info(f"resume from specified checkpoint {cfg.TRAIN.RESUME}.")
        epoch_start = load_checkpoint(cfg.TRAIN.RESUME, model, optimizer, scheduler, restart_train=True,
                                      rewrite=cfg.TRAIN.REWRITE_RESUME_MODEL)

    purge_step = epoch_start * len(dataloader)
    if not dist.is_initialized() or dist.get_rank() == 0:
        writer = SummaryWriter(os.path.join(cfg.LOG.DIR, cfg.LOG.TENSORBOARD_SUBDIR), purge_step=purge_step)
    else:
        writer = None

    loss_func = build_loss(cfg)
    meter = build_meter(cfg, name=cfg.TRAIN.METER, writer=writer, mode="train", purge_step=purge_step)

    if cfg.SYS.MULTIPROCESS:
        dist.barrier()

    for epoch in range(epoch_start, cfg.TRAIN.EPOCH):

        logger.info(f"Epoch {epoch + 1}/{cfg.TRAIN.EPOCH}")

        if "train_sampler" in dataloader:
            logger.info(f"set train sampler step to {epoch}")
            dataloader["train_sampler"].set_epoch(epoch)

        train_epoch(model=model,
                    dataloader=dataloader["train"],
                    optimizer=optimizer,
                    loss_func=loss_func,
                    meter=meter,
                    # only rank 0 process write tensorboard
                    writer=writer if not cfg.SYS.MULTIPROCESS or dist.get_rank() == 0 else None,
                    clip_norm=cfg.TRAIN.LOSS.CLIP_NORM,
                    epoch=epoch)

        # lr schedule for each epoch
        if cfg.TRAIN.SCHEDULER_WARMUP.ENABLE and \
                epoch < cfg.TRAIN.SCHEDULER_WARMUP.EPOCH and warmup_scheduler is not None:
            warmup_scheduler.step()
        elif scheduler is not None:
            scheduler.step()

        # eval
        if (epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0:
            logger.info(f"Eval for epoch {epoch + 1}")
            eval_epoch(model=model,
                       dataloader=dataloader["val"],
                       meter=meter,
                       writer=writer if not cfg.SYS.MULTIPROCESS or dist.get_rank() == 0 else None,
                       epoch=epoch)
        # save
        if (epoch + 1) % cfg.TRAIN.SAVE_PERIOD == 0:
            if not cfg.SYS.MULTIPROCESS or (cfg.SYS.MULTIPROCESS and dist.get_rank() == 0):
                # save the checkpoint in process with rank=0 if multiprocess is enabled
                save_checkpoint(ckpt_folder=os.path.join(cfg.LOG.DIR, cfg.LOG.CHECKPOINT_SUBDIR),
                                epoch=epoch + 1,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                config=cfg)

        if dist.is_initialized():
            dist.barrier()

    if writer is not None:
        writer.close()

    if torch.distributed.is_initialized():
        logger.info("Training process rank %s exit", dist.get_rank())
    else:
        logger.info("Training process exit")
