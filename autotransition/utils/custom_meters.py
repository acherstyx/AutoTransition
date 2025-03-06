# -*- coding: utf-8 -*-
# @Time    : 2021/12/22 11:36 AM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : custom_meters.py

import datetime
import itertools
import logging
import os
import random
from typing import *

import einops
import numpy as np
import torch
import torch.distributed as dist
from sklearn.manifold import TSNE

from .custom_loss import MaskedTripletLoss
from .meters import *

logger = logging.getLogger(__name__)


@METER_REGISTRY.register()
class ContrastiveFeatureMeter(MeterBase):

    def __init__(self, cfg, writer, mode, purge_step):
        super(ContrastiveFeatureMeter, self).__init__(cfg=cfg, writer=writer, mode=mode, purge_step=purge_step)
        self.feature = []
        self.label = []
        # accuracy
        self.accuracy_meter = AccuracyMeter(cfg, writer=writer, mode=mode, purge_step=purge_step)

    @torch.no_grad()
    def update(self,
               inputs: torch.Tensor,
               labels: torch.LongTensor,
               outputs: Tuple[torch.Tensor, torch.Tensor],
               n: int = None, global_step: int = None):
        output_feature, output_logits = outputs
        output_feature = output_feature.detach()
        output_logits = output_logits.detach()
        labels = labels.detach()
        assert inputs.shape[0] == labels.shape[0] == output_feature.shape[0] == output_logits.shape[0]  # batch size

        for cur_feat, cur_label in zip(output_feature, labels):
            self.feature.append(cur_feat.cpu())
            self.label.append(int(cur_label))

        self.accuracy_meter.update(inputs=None, labels=labels, outputs=output_logits, n=n, global_step=global_step)

        self.step += 1

    def _gather_multiple_gpu(self):
        logger.debug("gathering feature vectors from all process in the group")
        logger.debug("length before gathering: %s", len(self.feature))

        gathered_feature = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_feature, self.feature)
        self.feature = list(itertools.chain(*gathered_feature))

        gathered_label = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_label, self.label)
        self.label = list(itertools.chain(*gathered_label))

        logger.debug("the length of the gathered feature list: %s", len(self.feature))
        logger.debug("the length of the gathered label list: %s", len(self.label))

    @torch.no_grad()
    def summary(self, epoch):
        accuracy_summary_res = self.accuracy_meter.summary(epoch)

        if self.mode == "test":
            if dist.is_initialized():
                self._gather_multiple_gpu()
            if not dist.is_initialized() or dist.get_rank() == 0:
                self._generate_embedding()
                self._visualize_feature()  # t-SNE

        return accuracy_summary_res

    def reset(self):
        self.feature.clear()
        self.label.clear()
        self.accuracy_meter.reset()

    def _visualize_feature(self):
        import matplotlib.pyplot as plt
        num_total_points = 10000
        name_map = {i: (i + 30) % 31 for i in range(31)}

        t_sne = TSNE(n_components=2, n_iter=1000, metric="cosine")
        t_sne.fit(torch.stack(self.feature).cpu())

        color_map = {i: np.random.randint(0, 255, 3) / 255 for i in range(len(set(self.label)))}

        timestamp = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        save_path_drop_outliers = os.path.join(self.cfg.LOG.DIR,
                                               f"t-SNE-{timestamp}-drop_outliers.pdf")
        save_path_original = os.path.join(self.cfg.LOG.DIR,
                                          f"t-SNE-{timestamp}-original.pdf")

        x = t_sne.embedding_[:, 0]
        y = t_sne.embedding_[:, 1]

        plt.figure(dpi=600, figsize=(4, 4))
        if t_sne.embedding_.shape[1] == 2:
            ax: plt.Axes = plt.gca()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            handles = []
            labels = list(set(self.label))
            for select_label in labels:  # for each label, add scatter with different color and text
                cur_x = []
                cur_y = []
                cur_label = []
                cur_features = []
                for x_, y_, l_, feat in zip(x, y, self.label, self.feature):
                    if l_ == select_label:
                        cur_x.append(x_)
                        cur_y.append(y_)
                        cur_label.append(l_)
                        cur_features.append(feat.cpu().numpy())
                select = self._remove_outlier(cur_features)
                num_points = int(len(cur_label) / len(self.label) * num_total_points)
                cur_x = random.Random(222).choices(cur_x, k=num_points)
                cur_y = random.Random(222).choices(cur_y, k=num_points)
                cur_label = random.Random(222).choices(cur_label, k=num_points)
                select = random.Random(222).choices(select, k=num_points)
                filtered_x = list(itertools.compress(cur_x, select))
                filtered_y = list(itertools.compress(cur_y, select))
                filtered_label = list(itertools.compress(cur_label, select))
                color = [color_map[label] for label in filtered_label]
                handles.append(ax.scatter(filtered_x, filtered_y, c=color, s=20, alpha=0.4, edgecolors='none'))
                ax.text(sum(filtered_x) / len(filtered_x), sum(filtered_y) / len(filtered_y), name_map[select_label],
                        ha='center', va='center')
        else:
            raise NotImplementedError
        ax.set_axisbelow(True)
        ax.yaxis.grid(color="gainsboro", linestyle="dashed", linewidth=1)
        ax.xaxis.grid(color="gainsboro", linestyle="dashed", linewidth=1)
        plt.xticks(size=10)
        plt.yticks(size=10)
        plt.tight_layout()
        plt.savefig(save_path_drop_outliers)
        plt.close()
        logger.debug("t-SNE figure is saved to %s", save_path_drop_outliers)

        plt.figure(dpi=600, figsize=(4, 4))
        if t_sne.embedding_.shape[1] == 2:
            ax: plt.Axes = plt.gca()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            handles = []
            labels = list(set(self.label))
            for select_label in labels:  # for each label, add scatter with different color and text
                cur_x = []
                cur_y = []
                cur_label = []
                cur_features = []
                for x_, y_, l_, feat in zip(x, y, self.label, self.feature):
                    if l_ == select_label:
                        cur_x.append(x_)
                        cur_y.append(y_)
                        cur_label.append(l_)
                        cur_features.append(feat.cpu().numpy())
                select = self._remove_outlier(cur_features)
                num_points = int(len(cur_label) / len(self.label) * num_total_points)
                cur_x = random.Random(222).choices(cur_x, k=num_points)
                cur_y = random.Random(222).choices(cur_y, k=num_points)
                cur_label = random.Random(222).choices(cur_label, k=num_points)
                select = random.Random(222).choices(select, k=num_points)
                color = [color_map[label] for label in cur_label]
                handles.append(ax.scatter(cur_x, cur_y, c=color, s=20, alpha=0.5, edgecolors='none'))
                filtered_x = list(itertools.compress(cur_x, select))
                filtered_y = list(itertools.compress(cur_y, select))
                ax.text(sum(filtered_x) / len(filtered_x), sum(filtered_y) / len(filtered_y), name_map[select_label],
                        ha='center', va='center')
        else:
            raise NotImplementedError
        ax.set_axisbelow(True)
        ax.yaxis.grid(color="gainsboro", linestyle="dashed", linewidth=1)
        ax.xaxis.grid(color="gainsboro", linestyle="dashed", linewidth=1)
        plt.xticks(size=10)
        plt.yticks(size=10)
        plt.tight_layout()
        plt.savefig(save_path_original)
        plt.close()
        logger.debug("t-SNE figure is saved to %s", save_path_original)

    @staticmethod
    def _remove_outlier(data: List[List[Union[int, float]]]):
        data_arr = np.array(data)
        center = data_arr.mean(axis=0)
        dist = data_arr @ center
        select = np.abs(dist - dist.mean()) <= (3 * dist.std())
        logger.debug("remove outlier: (%s/%s)", np.sum(select), len(select))
        return select

    def _generate_embedding(self):
        assert len(self.feature) == len(self.label)
        num_class = max(list(set(self.label))) + 1
        d_feature = len(self.feature[0])

        embedding = [torch.zeros(d_feature) for _ in range(num_class)]

        for select_label in list(set(self.label)):
            cur_features = []
            for feat, label in zip(self.feature, self.label):
                if label == select_label:
                    cur_features.append(feat.cpu().numpy())
            select = self._remove_outlier(cur_features)
            filtered_features = list(itertools.compress(cur_features, select))
            filtered_features = torch.tensor(filtered_features)
            cur_embedding = torch.mean(filtered_features, dim=0)
            assert len(cur_embedding) == d_feature
            embedding[select_label] = cur_embedding

        embedding = torch.stack(embedding)

        save_path = os.path.join(self.cfg.LOG.DIR,
                                 f"embedding-{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pth")
        torch.save(embedding, save_path)
        logger.debug("embedding is saved to %s", save_path)

    def set_mode(self, mode: str):
        super(ContrastiveFeatureMeter, self).set_mode(mode)
        self.accuracy_meter.set_mode(mode)


@METER_REGISTRY.register()
class MaskedAccuracy(MeterBase):
    def __init__(self, cfg, writer, mode, purge_step):
        super(MaskedAccuracy, self).__init__(cfg=cfg, writer=writer, mode=mode, purge_step=purge_step)

        self.accuracy_meter = AccuracyMeter(cfg=cfg, writer=writer, mode=mode, purge_step=purge_step)

    @torch.no_grad()
    def update(self, inputs, labels, outputs, n=None, global_step=None):
        logits, mask = outputs
        mask = einops.rearrange(mask, "b n->(b n)")
        logits = einops.rearrange(logits, "b n cls->(b n) cls")[mask]
        labels = einops.rearrange(labels, "b n->(b n)")[mask]

        return self.accuracy_meter.update(inputs, labels, logits, global_step=global_step)

    @torch.no_grad()
    def summary(self, epoch):
        return self.accuracy_meter.summary(epoch=epoch)

    def reset(self):
        self.accuracy_meter.reset()

    def set_mode(self, mode: str):
        super(MaskedAccuracy, self).set_mode(mode)
        self.accuracy_meter.set_mode(mode)


@METER_REGISTRY.register()
class MaskedTripletMatchingAccuracy(MaskedAccuracy):
    def __init__(self, cfg, writer, mode, purge_step):
        super(MaskedTripletMatchingAccuracy, self).__init__(cfg=cfg, writer=writer, mode=mode, purge_step=purge_step)

        cfg = cfg.METER.MATCHING_ACCURACY

        assert cfg.DISTANCE in ["euclidean", "dot-production"], f"distance not support: {cfg.DISTANCE}"
        self.distance = cfg.DISTANCE

        self.squared = cfg.SQUARED

    @torch.no_grad()
    def update(self, inputs, labels, outputs, n=None, global_step=None):
        transition_embedding, prediction_embedding, label_mask = outputs

        bs, n, _ = prediction_embedding.shape
        prediction_embedding = einops.rearrange(prediction_embedding, "bs n dim_feat->(bs n) dim_feat")

        if self.distance == "euclidean":
            logits = - MaskedTripletLoss.euclidean_distance(transition_embedding=transition_embedding,
                                                            prediction_embedding=prediction_embedding,
                                                            squared=self.squared).T  # return shape: (bs*n, num_classes)
        elif self.distance == "dot-production":
            logits = - MaskedTripletLoss.dot_production_distance(transition_embedding=transition_embedding,
                                                                 prediction_embedding=prediction_embedding).T
        else:
            raise ValueError

        logits = einops.rearrange(logits, "(bs n) num_classes->bs n num_classes", bs=bs, n=n)

        super(MaskedTripletMatchingAccuracy, self).update(inputs=inputs,
                                                          labels=labels,
                                                          outputs=(logits, label_mask),
                                                          n=n, global_step=global_step)
