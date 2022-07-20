# -*- coding: utf-8 -*-
# @Time    : 2021/12/21 4:35 PM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : transition_embedding.py

import os
from typing import *
import logging
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .build import MODEL_REGISTRY
from .backbone import build_backbone

from torchvision.transforms.functional import normalize

logger = logging.getLogger(__name__)


@MODEL_REGISTRY.register()
class TransitionEmbedding(nn.Module):
    """
    train transition embedding
    """

    def __init__(self, cfg):
        super(TransitionEmbedding, self).__init__()

        backbone_cfg = cfg
        cfg = cfg.MODEL.TRANSITION_EMBEDDING

        if cfg.BACKBONE != "SlowFast":
            raise NotImplementedError
        self.backbone = build_backbone(backbone_cfg, cfg.BACKBONE)
        dim_feature = 2048 + 256
        d_embedding = 2048

        self.projection = nn.Linear(dim_feature, d_embedding)
        self.linear = nn.Linear(d_embedding, cfg.NUM_CLASSES)

    def forward(self, x):
        x = einops.rearrange(x, "b n c h w->b c n h w")
        x = x.float() / 255.0
        feat = self.backbone(x)
        feat = self.projection(feat)
        feat = F.normalize(feat, p=2.0, dim=1)
        cls = self.linear(feat)
        return feat, cls


@MODEL_REGISTRY.register()
class TransitionRecommendation(nn.Module):
    def __init__(self, cfg):
        super(TransitionRecommendation, self).__init__()

        # visual embedding
        self.backbone = build_backbone(cfg, cfg.MODEL.TRANSITION_TRANSFORMER.BACKBONE)
        for layer_name, param in self.backbone.named_parameters():
            logging.debug("backbone layer: %s", layer_name)
            if any(layer_name.startswith(stage) for stage in cfg.MODEL.TRANSITION_TRANSFORMER.FREEZE_STAGE):
                logger.debug("freeze: %s", layer_name)
                param.requires_grad = False
        dim_feature = self.backbone.d_feature
        self.visual_feature_projection = nn.Linear(dim_feature, cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_D_MODEL)
        # audio embedding
        self.audio_feature_projection = nn.Linear(100, cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_D_MODEL)

        # position & modal embeddings
        self.audio_position_embedding = nn.Parameter(
            torch.rand([100, 1, cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_D_MODEL]))
        self.visual_position_embedding = self.audio_position_embedding
        self.visual_modal_embedding = nn.Parameter(
            torch.rand([1, 1, cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_D_MODEL]))
        self.audio_modal_embedding = nn.Parameter(
            torch.rand([1, 1, cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_D_MODEL]))

        # multi-modal transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_D_MODEL,
            nhead=cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_N_HEAD)
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_NUM_LAYERS)

        # transition embedding
        assert cfg.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.PRETRAINED_EMBEDDING is not None and \
               os.path.exists(cfg.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.PRETRAINED_EMBEDDING), \
            f"embedding file is invalid: {cfg.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.PRETRAINED_EMBEDDING}"
        if not cfg.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.RANDOM_INITIALIZE:
            self.transition_embedding = nn.Parameter(
                torch.load(cfg.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.PRETRAINED_EMBEDDING), requires_grad=False)
        else:
            pretrained_embedding = torch.load(cfg.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.PRETRAINED_EMBEDDING)
            random_embedding = torch.randn_like(pretrained_embedding)
            random_embedding = torch.nn.functional.normalize(random_embedding, p=2.0, dim=1)
            self.transition_embedding = nn.Parameter(random_embedding, requires_grad=False)
        assert len(self.transition_embedding.shape) == 2

        # video embedding & transition embedding projection
        d_embedding = self.transition_embedding.shape[1]
        self.with_projection = cfg.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.TRANSITION_EMBEDDING_PROJECTION
        if self.with_projection:
            self.transition_embedding_projection = nn.Linear(d_embedding,
                                                             cfg.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.D_COMMON_SPACE)
            self.video_embedding_projection = nn.Linear(cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_D_MODEL * 4,
                                                        cfg.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.D_COMMON_SPACE)
        else:
            self.video_embedding_projection = nn.Linear(cfg.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_D_MODEL * 4,
                                                        d_embedding)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        video, music_feature, label_mask, music_mask = x

        video = video.float() / 255
        batch_size = video.shape[0]
        frame_per_clip = video.shape[3]
        video_mask = ~einops.repeat(label_mask, "b n->b (n 2)")
        music_mask = ~music_mask

        # check
        assert video.shape[1] == music_feature.shape[1], "visual and audio feature should in the same length"
        video = einops.rearrange(video, "b l c n h w->(b l n) c h w")
        video = normalize(video, [0.45, 0.45, 0.45], [0.225, 0.225, 0.225])
        video = einops.rearrange(video, "(b l n) c h w->(b l) c n h w", b=batch_size, n=frame_per_clip)

        # video -> video token
        video_token = self.backbone(video)
        video_token = einops.rearrange(video_token, "(b l) c->l b c", b=batch_size)
        video_token = self.visual_feature_projection(video_token)
        video_token = video_token + \
                      self.visual_position_embedding[:video_token.shape[0], :, :] + self.visual_modal_embedding

        # music feature -> music token
        music_token = einops.rearrange(music_feature, "b l c->l b c")
        music_token = self.audio_feature_projection(music_token)
        music_token = music_token + \
                      self.audio_position_embedding[:music_token.shape[0], :, :] + self.audio_modal_embedding

        tokens = torch.cat([video_token, music_token], dim=0)
        padding_mask = torch.cat([video_mask, music_mask], dim=1)
        assert tokens.shape[0] == padding_mask.shape[1], f"shape not match: {tokens.shape} vs {padding_mask.shape}"
        assert tokens.shape[1] == padding_mask.shape[0], f"shape not match: {tokens.shape} vs {padding_mask.shape}"

        tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        tokens = einops.rearrange(tokens, "(modal n_transition two) b c->b (modal two c) n_transition", modal=2, two=2)
        tokens = einops.rearrange(tokens, "b c n->b n c")

        prediction_embedding = self.video_embedding_projection(tokens)
        if self.with_projection:
            transition_embedding = self.transition_embedding_projection(self.transition_embedding)
        else:
            transition_embedding = self.transition_embedding

        return transition_embedding, prediction_embedding, label_mask
