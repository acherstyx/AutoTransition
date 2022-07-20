# -*- coding: utf-8 -*-
# @Time    : 2021/12/23 2:05 PM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : custom_loss.py

import logging
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from .loss import *


logger = logging.getLogger(__name__)


@LOSS_REGISTRY.register()
class ClassificationLoss(LossBase):

    def forward(self, outputs, labels):
        return self.similarity_and_classification_loss(outputs, labels)

    @staticmethod
    def similarity_and_classification_loss(outputs: Tuple[torch.Tensor, torch.Tensor], labels: torch.Tensor):
        feature_vector, classification_logits = outputs  # bs*dim_feature, bs*num_classes

        cross_entropy_loss = F.cross_entropy(classification_logits, labels)

        logger.debug("cross entropy: %s", cross_entropy_loss.detach().cpu().numpy())

        return cross_entropy_loss


@LOSS_REGISTRY.register()
class MaskedTripletLoss(LossBase):

    def __init__(self, cfg):
        super(MaskedTripletLoss, self).__init__(cfg=cfg)

        cfg = cfg.LOSS.TRIPLET_LOSS

        self.squared = cfg.SQUARED
        self.margin = cfg.MARGIN
        assert cfg.DISTANCE in ["euclidean", "dot-production"], f"distance not support: {cfg.DISTANCE}"
        self.distance = cfg.DISTANCE

    @staticmethod
    def euclidean_distance(transition_embedding: torch.Tensor,
                           prediction_embedding: torch.Tensor,
                           squared: bool = False):
        """
        distance between transition embedding and prediction embedding
        :param transition_embedding: (num_classes, dim_feat)
        :param prediction_embedding: (batch_size/n, dim_feat)
        :param squared:
        :return: (num_classes, batch_size/n) pairwise distance matrix
        """

        assert transition_embedding.shape[1] == prediction_embedding.shape[1], \
            f"feature dim not match: {transition_embedding.shape[1]} vs {prediction_embedding.shape[1]}"

        distances = torch.sum((transition_embedding.unsqueeze(1) - prediction_embedding.unsqueeze(0)) ** 2, dim=-1)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = (distances == torch.tensor(0.0, dtype=torch.float).to(distances.device)).float()
            distances = distances + mask * 1e-16

            distances = torch.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        return distances

    @staticmethod
    def dot_production_distance(transition_embedding: torch.Tensor,
                                prediction_embedding: torch.Tensor):
        """
        distance between transition embedding and prediction embedding
        :param transition_embedding: (num_classes, dim_feat)
        :param prediction_embedding: (batch_size/n, dim_feat)
        :return: (num_classes, batch_size/n) pairwise distance matrix
        """

        assert transition_embedding.shape[1] == prediction_embedding.shape[1], \
            f"feature dim not match: {transition_embedding.shape[1]} vs {prediction_embedding.shape[1]}"

        distances = transition_embedding @ prediction_embedding.T

        return distances

    @staticmethod
    def _get_triplet_mask(labels: torch.LongTensor, num_classes: int, mask: torch.BoolTensor = None):
        """
        valid triplet mask + label mask
        :param labels: (batch_size/n, )
        :param num_classes: integer
        :param mask: (batch_size/n, ), padding label mask, False for padding sample, True for no padding samples
        """
        n = labels.size(0)  # number of samples in current batch
        device = labels.device
        i = labels.view(-1, 1, 1).expand(n, num_classes, num_classes)
        j = torch.range(0, num_classes - 1, dtype=torch.long, device=device).view(1, -1, 1).expand(n, num_classes,
                                                                                                   num_classes)
        k = torch.range(0, num_classes - 1, dtype=torch.long, device=device).view(1, 1, -1).expand(n, num_classes,
                                                                                                   num_classes)
        triplet_mask = torch.logical_and(i == j, j != k)

        if mask is not None:
            triplet_padding_mask = mask.view(-1, 1, 1)
            triplet_mask = torch.logical_and(triplet_mask, triplet_padding_mask)
        return triplet_mask

    def forward(self, outputs, labels):
        """
        calculate loss
        :param outputs:
            transition embedding: (num_classes, dim_feat)
            prediction embedding: (batch_size, n, dim_feat)
            mask: (batch_size, n)
        :param labels:
        """
        transition_embedding, prediction_embedding, mask = outputs
        prediction_embedding = einops.rearrange(prediction_embedding, "bs n dim_feat->(bs n) dim_feat")
        mask = einops.rearrange(mask, "bs n->(bs n)")
        labels = einops.rearrange(labels, "bs n->(bs n)")

        num_classes = transition_embedding.size(0)

        if self.distance == "euclidean":
            pairwise_dist = self.euclidean_distance(transition_embedding=transition_embedding,
                                                    prediction_embedding=prediction_embedding,
                                                    squared=self.squared)
        elif self.distance == "dot-production":
            pairwise_dist = self.dot_production_distance(transition_embedding=transition_embedding,
                                                         prediction_embedding=prediction_embedding)
        else:
            raise ValueError

        pairwise_dist = pairwise_dist.T  # (num_classes, batch_size/n) -> (batch_size/n, num_classes)

        anchor_positive_dist = torch.unsqueeze(pairwise_dist, 2)  # batch_size, num_classes, 1
        anchor_negative_dist = torch.unsqueeze(pairwise_dist, 1)  # batch_size, 1, num_classes

        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin

        triplet_mask = self._get_triplet_mask(labels, num_classes=num_classes, mask=mask)
        triplet_mask = triplet_mask.float()
        triplet_loss = triplet_mask * triplet_loss

        triplet_loss = torch.max(triplet_loss, torch.tensor(0.0, dtype=torch.float).to(triplet_loss.device))

        valid_triplets = (triplet_loss > 1e-16).float()
        num_positive_triplets = torch.sum(valid_triplets)

        triplet_loss = torch.sum(triplet_loss) / (num_positive_triplets + 1e-16)

        return triplet_loss
