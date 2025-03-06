# -*- coding: utf-8 -*-
# @Time    : 2021/12/22 11:33 AM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : meters.py

import datetime
import logging
import os
from typing import *

import numpy as np
import torch
import torch.distributed as dist
from fvcore.common.config import CfgNode
from fvcore.common.registry import Registry
from torch.utils.tensorboard import SummaryWriter

from .distributed import gather_object_multiple_gpu

__all__ = ["METER_REGISTRY", "MeterBase", "AccuracyMeter", "DummyMeter", "build_meter"]

METER_REGISTRY = Registry("METER")
logger = logging.getLogger(__name__)


class MeterBase(object):
    """
    Interface
    """

    def __init__(self, cfg: CfgNode, writer: SummaryWriter, mode: str, purge_step: int):
        """
        build meter
        :param cfg: config
        :param writer: initialized tensorboard summary writer
        :param mode: train, val or test mode, for tagging tensorboard, etc.
        :param purge_step: initial step, for recover
        """
        assert mode in ["train", "test", "val"], f"mode is invalid: {mode}"
        assert purge_step >= 0, "step should greater than zero"
        self.cfg = cfg.clone()
        self.writer = writer
        self.mode = mode
        self.step = purge_step  # +1 when update

    def set_mode(self, mode: str):
        assert mode in ["train", "test", "val"], f"mode is invalid: {mode}"
        self.mode = mode

    @torch.no_grad()
    def update(self, inputs, labels, outputs, n=None, global_step=None):
        """
        call on each step
        update inner status based on the input
        :param inputs: the dataloader output
        :param labels: the dataloader output
        :param outputs: the model output
        :param n: number of steps
        :param global_step: global step, use `self.step` if step is None
        """
        raise NotImplementedError

    @torch.no_grad()
    def summary(self, epoch):
        """
        call at the end of the epoch
        """
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class DummyMeter(MeterBase):

    def update(self, inputs, labels, outputs, n=None, global_step=None):
        pass

    def summary(self, epoch):
        pass

    def reset(self):
        pass


@METER_REGISTRY.register()
class AccuracyMeter(MeterBase):

    def __init__(self, cfg: CfgNode, writer: SummaryWriter, mode: str, purge_step: int):
        super(AccuracyMeter, self).__init__(cfg=cfg, writer=writer, mode=mode, purge_step=purge_step)

        self._top1 = []
        self._top5 = []
        self._rank = []
        self._batch_size = []

        self._label = []
        self._predict = []

    @staticmethod
    @torch.no_grad()
    def topk_accuracy(logits: torch.Tensor, target: torch.LongTensor, topk=(1, 5), verbose=False):
        assert len(logits.shape) == 2
        assert len(target.shape) == 1
        assert logits.shape[0] == target.shape[0]

        max_k = max(topk)
        batch_size = target.size(0)

        _, pred = logits.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            cur_acc = correct_k.mul_(100.0 / batch_size)[0]
            res.append(cur_acc)

        # is distributed is enabled, gather accuracy from other GPU device and average it
        if dist.is_initialized():
            for cur_acc in res:
                dist.all_reduce(cur_acc)
            res = [cur_acc / dist.get_world_size() for cur_acc in res]
        # return the result as a dict, convert torch tensor to float value
        ret = {f"top{k}": v.cpu().numpy().item() for k, v in zip(topk, res)}

        if verbose:  # debug log
            logger.debug("accuracy metric (predict, gt): %s",
                         list(zip(torch.argmax(logits, dim=-1).cpu().numpy(), target.cpu().numpy())))
            if dist.is_initialized():
                logger.debug(f"accuracy (rank {dist.get_rank()}): %s", ret)
            else:
                logger.debug("accuracy (no rank): %s", ret)
        return ret

    @staticmethod
    @torch.no_grad()
    def rank(logits: torch.Tensor, target: torch.LongTensor, verbose=False):
        """
        get rank of the target index in the prediction output
        :param logits: prediction with shape (batch_size, num_class)
        :param target: label with shape (batch_size)
        :param verbose: show debug log
        """
        assert len(logits.shape) == 2
        assert len(target.shape) == 1
        assert logits.shape[0] == target.shape[0]
        # size
        num_class = logits.size(1)
        batch_size = logits.size(0)
        _, pred = logits.topk(num_class, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        assert torch.sum(correct) == batch_size, "target is incorrect (larger than num_class in logits)"
        _, rank = correct.float().topk(1, dim=0)

        rank = torch.mean(rank.float())

        if dist.is_initialized():
            logger.debug("rank before averaging (rank %s): %s", dist.get_rank(), rank)
            dist.all_reduce(rank)
            rank = rank / dist.get_world_size()
        else:
            logger.debug("rank before averaging: %s", rank)

        if verbose:
            logger.debug("rank in each sample: %s", rank.detach().cpu().numpy())

        return rank.detach().cpu().numpy().item()

    @torch.no_grad()
    def update(self, inputs, labels, outputs, n=None, global_step=None):
        """
        update for each step
        :param inputs: not used in this meter
        :param labels: 1D LongTensor for classification
        :param outputs: 2D logits with shape (BATCH_SIZE, NUM_CLASSES)
        :param n: set to None to use batch size for averaging across epoch
        :param global_step: specify global step manually
        """
        accuracy_result = self.topk_accuracy(logits=outputs, target=labels, topk=(1, 5), verbose=True)
        if n is None:
            n = labels.shape[0]
        assert isinstance(n, int), f"expect n to be an integer, get {type(n)}"
        self._top1.append(accuracy_result["top1"])
        self._top5.append(accuracy_result["top5"])
        self._batch_size.append(n)

        rank = self.rank(outputs, labels, verbose=True)
        self._rank.append(rank)

        for cur_label in labels:
            self._label.append(cur_label.item())
        for cur_logits in outputs:
            assert len(cur_logits.shape) == 1, f"logits should be 1 dim tensor, received {cur_logits.shape}"
            pred = torch.argmax(cur_logits).item()
            self._predict.append(pred)
        assert len(self._label) == len(self._predict)

        if global_step is None:
            global_step = self.step
        # write tensorboard
        if self.mode in ["train"] and self.writer is not None and (not dist.is_initialized() or dist.get_rank() == 0):
            self.writer.add_scalars(f"{self.mode}/accuracy", accuracy_result, global_step=global_step)
            self.writer.add_scalar(f"{self.mode}/rank", rank, global_step=global_step)

        self.step += 1

    @torch.no_grad()
    def summary(self, epoch):
        # average accuracy
        acc_top1 = np.array(self._top1)
        acc_top5 = np.array(self._top5)
        batch_size = np.array(self._batch_size)
        acc_top1 = np.sum(acc_top1 * batch_size) / np.sum(batch_size)
        acc_top5 = np.sum(acc_top5 * batch_size) / np.sum(batch_size)

        rank = np.array(self._rank)
        avg_rank = np.sum(rank * batch_size) / np.sum(batch_size)

        if dist.is_initialized():
            labels = np.array(gather_object_multiple_gpu(self._label))
            predicts = np.array(gather_object_multiple_gpu(self._predict))
        else:
            labels = np.array(self._label)
            predicts = np.array(self._predict)
        correct = (labels == predicts).astype(int)
        individual_labels = list(set(labels))
        accuracy_per_class = {}
        for cur_label in individual_labels:
            correct_selected = correct[labels == cur_label]
            accuracy_per_class[cur_label] = correct_selected.sum() / len(correct_selected)
        logger.debug("accuracy for each class: %s", accuracy_per_class)
        if not dist.is_initialized() or dist.get_rank() == 0:
            img = self.visualize_per_class_accuracy(accuracy_per_class, acc_top1)
            self.writer.add_image(f"{self.mode}/accuracy_epoch", img, global_step=epoch, dataformats="HWC")

        acc_res = {"top1": acc_top1, "top5": acc_top5}
        if self.writer is not None and (not dist.is_initialized() or dist.get_rank() == 0) and epoch is not None:
            self.writer.add_scalars(f"{self.mode}/accuracy_epoch", acc_res, global_step=epoch)
            self.writer.add_scalar(f"{self.mode}/rank_epoch", avg_rank, global_step=epoch)
        logger.debug("accuracy summary (epoch %s): %s", epoch, acc_res)
        logger.debug("rank summary (epoch %s): %s", epoch, avg_rank)

        return {"top1": acc_top1,
                "top5": acc_top5,
                "rank": avg_rank}

    def reset(self):
        self._top1.clear()
        self._top5.clear()
        self._batch_size.clear()
        self._label.clear()
        self._predict.clear()
        self._rank.clear()

    def visualize_per_class_accuracy(self, accuracy_dict: Dict[Any, str], average_acc: float):
        import matplotlib.pyplot as plt

        labels = [k for k, v in accuracy_dict.items()]
        acc = [v * 100 for k, v in accuracy_dict.items()]

        save_path = os.path.join(self.cfg.LOG.DIR,
                                 f"Accuracy-{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.png")

        ax: plt.Axes = plt.gca()
        ax.bar(labels, acc)
        ax.set_xlabel("Label")
        ax.set_ylabel("Accuracy(%)")
        ax.set_ylim(0.0, 100.0)
        ax.axhline(average_acc, color="r", label="average", linestyle=":")
        plt.savefig(save_path)
        logger.debug("Accuracy for each class is saved to %s", save_path)
        plt.close()
        img = plt.imread(save_path)
        return img


def build_meter(cfg: CfgNode, name: str, writer: SummaryWriter, mode: str, purge_step: int):
    if name is not None and name:
        logger.debug("Use meter: %s", name)
        return METER_REGISTRY.get(name)(cfg=cfg, writer=writer, mode=mode, purge_step=purge_step)
    else:
        logger.warning("Meter is not specified, will use dummy meter!")
        return DummyMeter(cfg=cfg, writer=writer, mode=mode, purge_step=purge_step)  # no metric
