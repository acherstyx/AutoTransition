# -*- coding: utf-8 -*-
# @Time    : 2021/11/29 11:23 AM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : custom_config.py

from yacs.config import CfgNode as CN

from autotransition.model.backbone.slowfast.config.defaults import get_cfg as get_slowfast_config


# noinspection DuplicatedCode
def add_custom_config(_C: CN) -> None:
    # additional config for model, dataset, etc.

    # ====================
    #       dataset
    # ====================

    # transition classification, for transition embedding training with contrastive learning
    _C.DATASET.TRANSITION_CLASSIFICATION = CN()
    _C.DATASET.TRANSITION_CLASSIFICATION.JSON_ANNOTATION = "./dataset/special_effect/annotation.json"
    _C.DATASET.TRANSITION_CLASSIFICATION.TEMPLATE_ROOT = "./dataset/special_effect/template"
    _C.DATASET.TRANSITION_CLASSIFICATION.NUM_CLASSES = 30
    _C.DATASET.TRANSITION_CLASSIFICATION.WITH_DIRECT_CUT = False
    _C.DATASET.TRANSITION_CLASSIFICATION.BALANCE = True
    _C.DATASET.TRANSITION_CLASSIFICATION.BALANCE_SHUFFLE_SEED = 222
    _C.DATASET.TRANSITION_CLASSIFICATION.SIZE = (224, 224)
    _C.DATASET.TRANSITION_CLASSIFICATION.FRAME_PER_CLIP = 32

    # transition
    # baseline dataset
    _C.DATASET.TRANSITION_DATASET = CN()
    _C.DATASET.TRANSITION_DATASET.JSON_ANNOTATION = "./dataset/special_effect/annotation.json"
    _C.DATASET.TRANSITION_DATASET.TEMPLATE_ROOT = "./dataset/special_effect/template"
    _C.DATASET.TRANSITION_DATASET.SIZE = (224, 224)
    _C.DATASET.TRANSITION_DATASET.FRAME_PER_CLIP = 16
    _C.DATASET.TRANSITION_DATASET.STEP = 2
    # single version
    _C.DATASET.TRANSITION_DATASET.SINGLE = CN()
    # sequence version
    _C.DATASET.TRANSITION_DATASET.SEQUENCE = CN()
    _C.DATASET.TRANSITION_DATASET.SEQUENCE.MAX_SEQUENCE_LEN = 10
    _C.DATASET.TRANSITION_DATASET.SEQUENCE.NUM_CLASSES = 30
    _C.DATASET.TRANSITION_DATASET.SEQUENCE.WITH_DIRECT_CUT = False

    # ====================
    #        model
    # ====================

    # transition embedding
    _C.MODEL.TRANSITION_EMBEDDING = CN()
    _C.MODEL.TRANSITION_EMBEDDING.NUM_CLASSES = 30
    _C.MODEL.TRANSITION_EMBEDDING.BACKBONE = "SlowFast"

    # transition recommendation
    # base config
    _C.MODEL.TRANSITION_TRANSFORMER = CN()
    _C.MODEL.TRANSITION_TRANSFORMER.CLASS_NUM = 30
    _C.MODEL.TRANSITION_TRANSFORMER.BACKBONE = "SlowFast"
    _C.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_D_MODEL = 2048
    _C.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_N_HEAD = 8
    _C.MODEL.TRANSITION_TRANSFORMER.TRANSFORMER_NUM_LAYERS = 2
    _C.MODEL.TRANSITION_TRANSFORMER.FREEZE_STAGE = ["s1", "s2", "s3"]  # slowfast stage s1 - s5
    # embedding matching model
    _C.MODEL.TRANSITION_TRANSFORMER.EMBEDDING = CN()
    _C.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.D_COMMON_SPACE = 2048
    _C.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.PRETRAINED_EMBEDDING = "./model_zoo/pretrain/" \
                                                                     "transition_embedding_vector.pth"
    _C.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.RANDOM_INITIALIZE = False
    _C.MODEL.TRANSITION_TRANSFORMER.EMBEDDING.TRANSITION_EMBEDDING_PROJECTION = True

    # ====================
    # backbone
    # ====================

    # add slowfast config
    _C.BACKBONE.SLOWFAST = get_slowfast_config()  # check other config in model/backbone/slowfast/config/defaults.py
    _C.BACKBONE.SLOWFAST.CHECKPOINT = "./model_zoo/slowfast/SLOWFAST_8x8_R50.pkl"
    _C.BACKBONE.SLOWFAST.ADDITIONAL_CONFIG = "./model/backbone/slowfast/configs/Kinetics/SLOWFAST_8x8_R50.yaml"

    # ====================
    # loss
    # ====================
    _C.LOSS.TRIPLET_LOSS = CN()
    _C.LOSS.TRIPLET_LOSS.MARGIN = 0.3
    _C.LOSS.TRIPLET_LOSS.DISTANCE = 'euclidean'  # 'euclidean' or 'dot-production'
    _C.LOSS.TRIPLET_LOSS.SQUARED = False  # only valid for euclidean distance

    # ====================
    # meter
    # ====================
    _C.METER.MATCHING_ACCURACY = CN()
    _C.METER.MATCHING_ACCURACY.DISTANCE = 'euclidean'  # 'euclidean' or 'dot-production'
    _C.METER.MATCHING_ACCURACY.SQUARED = False  # only valid for euclidean distance
