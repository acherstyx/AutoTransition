# -*- coding: utf-8 -*-
# @Time    : 2021/12/7 5:53 PM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : transition_dataset.py

import copy
import itertools
import os
import json

import einops
import numpy as np
import torch

from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision.transforms import *
from fractions import Fraction
from yacs.config import CfgNode

from .io import *
from .build import DATASET_REGISTRY
from utils import try_cache_load, cache_save

import logging
import warnings
from typing import *

logger = logging.getLogger(__name__)

logging.getLogger("numba").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')


def _n_h_w_c2n_c_h_w(x):
    return einops.rearrange(x, "n h w c->n c h w")


def _n_c_h_w2c_n_h_w(x):
    return einops.rearrange(x, "n c h w->c n h w")


@DATASET_REGISTRY.register()
class TransitionDataset(Dataset):
    def __init__(self, cfg, mode):
        super(TransitionDataset, self).__init__()
        assert mode in ("train", "test", "val")
        cfg = cfg.DATASET.TRANSITION_DATASET

        self.frame_num = cfg.FRAME_PER_CLIP
        self.step = cfg.STEP
        self.templates_root_dir = cfg.TEMPLATE_ROOT

        # load annotation
        with open(cfg.JSON_ANNOTATION, "r") as f:
            json_annotation: dict = json.load(f)
        if mode in ["train"]:
            json_annotation["templates"] = json_annotation["templates"]["train"]
        else:
            json_annotation["templates"] = json_annotation["templates"]["test"]

        if mode in ["train"]:
            self.transform = Compose([
                _n_h_w_c2n_c_h_w,
                Resize(min(cfg.SIZE)),
                RandomResizedCrop(cfg.SIZE, scale=(0.8, 1.2)),
                _n_c_h_w2c_n_h_w
            ])
        else:
            self.transform = Compose([
                _n_h_w_c2n_c_h_w,
                Resize(min(cfg.SIZE)),
                CenterCrop(cfg.SIZE),
                _n_c_h_w2c_n_h_w
            ])

        self.tid_list = sorted([tid for tid in list(set(list(json_annotation["templates"].keys())))
                                if 0 < len(json_annotation["templates"][tid]["transition"]) <= 100])
        self.video_path_list = [os.path.join(cfg.TEMPLATE_ROOT, tid, f"{tid[-4:]}out_video.mp4")
                                for tid in self.tid_list]
        self.transitions_list = [[t for t in json_annotation["templates"][tid]["transition"]] for tid in self.tid_list]
        assert len(self.tid_list) == len(self.video_path_list) == len(self.transitions_list)

        # remove overlapped transition in the dataset
        no_overlap_transition_list = []
        for ts in self.transitions_list:
            keep_list = []
            pre_end = 0
            for t in ts:
                assert t["start"] <= t["end"], f"transition should not be empty: {t}"
                if t["start"] > pre_end:
                    keep_list.append(t)
                    pre_end = t["end"]
            assert keep_list, "no transition in this template"
            no_overlap_transition_list.append(keep_list)
        # check if the duration of the transitions is still overlap
        for ts in no_overlap_transition_list:
            pre_end = 0
            for t in ts:
                assert pre_end < t["start"], ts
                pre_end = t["end"]
        self.transitions_list = no_overlap_transition_list  # update transitions list
        # check the result and write some log info
        assert len(self.tid_list) == len(self.video_path_list) == len(self.transitions_list)
        logger.debug("the number of templates in the transition dataset (mode=%s) is: %s", mode, len(self.tid_list))
        logger.debug("the number of transitions in the transition dataset (mode=%s) is: %s",
                     mode, len(list(itertools.chain(*self.transitions_list))))
        logger.debug("transitions per template: %s",
                     np.array([len(t_list) for t_list in self.transitions_list]).sum() / len(self.transitions_list))
        logger.debug("max: %s", np.array([len(t_list) for t_list in self.transitions_list]).max())

        # compute pts or load pts from cache
        cache_file = os.path.join(cfg.TEMPLATE_ROOT, f"pts_cache_{mode}.pth")
        self.pts_list = try_cache_load(cache_file, match=self.video_path_list)
        if self.pts_list is None:  # no cache or load failed
            self.pts_list, _ = compute_pts_tasker(self.video_path_list, use_frame=True, enable_process_bar=False)
            # convert to tuple for faster serialize speed
            self.pts_list = [[(frac.numerator, frac.denominator) for frac in frac_list]
                             for frac_list in self.pts_list]
        assert len(self.pts_list) == len(self.tid_list)
        if dist.is_initialized():
            dist.barrier()
        cache_save(cache_file, self.pts_list, match=self.video_path_list)

        self.class_dict = {k: index for index, k in enumerate(list(json_annotation["statistic"]["transition"].keys()))}

        # do some statistic about long-tailed classification
        self.class_statistic = json_annotation["statistic"]["transition"]
        logger.debug("class statistic (mode %s): %s", mode, self.class_statistic)
        logger.debug("weight info: %s", [v for k, v in self.class_statistic.items()])
        weight = np.array([v for k, v in self.class_statistic.items()])
        logger.debug("ratio: %s", {k: round(v / weight.sum() * 100, 2) for k, v in self.class_statistic.items()})

    @staticmethod
    def separate_transition(pts_list: List[Fraction], transitions: List[dict], index: int):
        transition_cur = transitions[index]
        if index > 0:
            transition_pre = transitions[index - 1]
        else:
            transition_pre = None
        if index < len(transitions) - 1:
            transition_post = transitions[index + 1]
        else:
            transition_post = None

        # pre clip start and end
        if transition_pre is not None:
            pre_clip_start = transition_pre["end"] / 1000
        else:
            pre_clip_start = 0.0
        pre_clip_end = transition_cur["start"] / 1000
        # post clip start and end
        post_clip_start = transition_cur["end"] / 1000
        if transition_post is not None:
            post_clip_end = transition_post["start"] / 1000
        else:
            post_clip_end = np.inf  # no upper bound
        # check
        assert pre_clip_start <= pre_clip_end <= post_clip_start <= post_clip_end

        pts_pre = []
        pts_post = []

        for pts in pts_list:
            if pre_clip_start < float(pts) < pre_clip_end:
                pts_pre.append(pts)
            elif post_clip_start < float(pts) < post_clip_end:
                pts_post.append(pts)

        return pts_pre, pts_post

    def __getitem__(self, item):
        raise NotImplementedError


@DATASET_REGISTRY.register()
class MultimodalTransitionRecommendationDataset(TransitionDataset):

    def __init__(self, cfg, mode):
        super(MultimodalTransitionRecommendationDataset, self).__init__(cfg, mode)
        cfg = cfg.DATASET.TRANSITION_DATASET.SEQUENCE

        self.max_sequence_len = cfg.MAX_SEQUENCE_LEN

        if cfg.WITH_DIRECT_CUT:
            new_class_dict = {t: i for i, t in enumerate(list(self.class_statistic)[:cfg.NUM_CLASSES])}
        else:
            new_class_dict = {t: i for i, t in enumerate(list(self.class_statistic)[1:cfg.NUM_CLASSES + 1])}
        logger.debug("New class dict: %s", new_class_dict)

        # remove other class
        new_tid_list = []
        new_video_path_list = []
        new_transitions_list = []
        self.transitions_full_list = []
        self.transitions_keep2full_index_list = []
        for tid, video_path, transitions in zip(self.tid_list, self.video_path_list, self.transitions_list):
            new_transitions = []
            transitions_full = []
            transitions_keep_index = []
            for i, t in enumerate(transitions):
                transitions_full.append(t)
                if t["name"] in new_class_dict:
                    new_transitions.append(t)
                    transitions_keep_index.append(i)
            if new_transitions:
                new_tid_list.append(tid)
                new_video_path_list.append(video_path)
                new_transitions_list.append(new_transitions)
                self.transitions_full_list.append(transitions_full)
                self.transitions_keep2full_index_list.append(transitions_keep_index)

        self.class_dict = new_class_dict
        self.tid_list = new_tid_list
        self.video_path_list = new_video_path_list
        self.transitions_list = new_transitions_list

    def __len__(self):
        return len(self.tid_list)

    def get_sequence_transition(self, template_index):
        tid = self.tid_list[template_index]
        transitions = self.transitions_list[template_index]
        pts_list = [Fraction(pts[0], pts[1]) for pts in self.pts_list[template_index]]
        video_file = os.path.join(self.templates_root_dir, tid, f"{tid[-4:]}out_video.mp4")
        transitions_full = self.transitions_full_list[template_index]
        transitions_keep2full_index = self.transitions_keep2full_index_list[template_index]

        video_pts = []
        for transition_index in range(min(len(transitions), self.max_sequence_len)):
            cur_transition = transitions[transition_index]
            keep2full_index = transitions_keep2full_index[transition_index]
            assert cur_transition == transitions_full[keep2full_index]

            pts_pre, pts_post = self.separate_transition(pts_list, transitions_full, keep2full_index)
            video_pts.append((pts_pre, pts_post))

        pts_set = [pts_list[0], pts_list[-1]]
        for pre, post in video_pts:
            pts_set.extend(pre)
            pts_set.extend(post)
        pts_set = list(set(pts_set))
        pts_dict = {}
        for pts in pts_set:
            pts_dict[pts] = read_video_frame(video_file, pts)

        video_seq = []
        label_seq = []
        for (pts_pre, pts_post), transition_index in zip(video_pts,
                                                         range(min(len(transitions), self.max_sequence_len))):
            cur_transition = transitions[transition_index]
            pts_pre = pts_pre[-self.frame_num * self.step::self.step]
            pts_post = pts_post[:self.frame_num * self.step:self.step]

            if 0 < len(pts_pre) < self.frame_num:
                # append
                pts_pre = [pts_pre[0], ] * (self.frame_num - len(pts_pre)) + pts_pre
            elif len(pts_pre) == 0:
                logger.debug("no pre frame for template %s, transition annotation %s",
                             tid, transitions)
                pts_pre = [pts_list[0], ] * self.frame_num
            assert len(pts_pre) == self.frame_num

            if 0 < len(pts_post) < self.frame_num:
                pts_post = pts_post + [pts_post[-1], ] * (self.frame_num - len(pts_post))
            elif len(pts_post) == 0:
                logger.debug("no post frame for template %s, transition annotation %s",
                             tid, transitions)
                pts_post = [pts_list[-1], ] * self.frame_num
            assert len(pts_post) == self.frame_num

            pre_frame = torch.stack([pts_dict[pts] for pts in pts_pre])
            post_frame = torch.stack([pts_dict[pts] for pts in pts_post])

            pre_frame = self.transform(pre_frame)
            post_frame = self.transform(post_frame)
            label = torch.tensor(self.class_dict[cur_transition["name"]], dtype=torch.long)
            video_seq.append(pre_frame)
            video_seq.append(post_frame)
            label_seq.append(label)
        assert len(video_seq) == 2 * len(label_seq)
        assert len(label_seq) <= self.max_sequence_len
        return torch.stack(video_seq), torch.stack(label_seq)

    def __getitem__(self, index):
        try:
            video_res, label_list = self.get_sequence_transition(index)
            cur_tid = self.tid_list[index]
            transitions = self.transitions_list[index]
            transitions = transitions[:len(label_list)]  # slice
            assert len(transitions) == len(label_list) and len(transitions) * 2 == len(video_res)
            for t, l in zip(transitions, label_list):
                assert self.class_dict[t["name"]] == l, f"label not match: {self.class_dict[t['name']]} vs {l}"
            annotation_time_list = []
            for t in transitions:
                annotation_time_list.append(t["start"])
                annotation_time_list.append(t["end"])

            local_feature_file = os.path.join(self.templates_root_dir, cur_tid, "audio_sync_local_feature.json")

            with open(local_feature_file, "r") as f:
                local_feature_list_source = [(float(k), torch.tensor(v, dtype=torch.float))
                                             for k, v in json.load(f)["embed"].items()]

            local_feature_list_source = sorted(local_feature_list_source, key=lambda x: x[0])  # sort by time
            if len(local_feature_list_source) != len(annotation_time_list):
                # match transition time with music feature
                local_feature_list = []
                for at_time in annotation_time_list:
                    min_diff = np.inf  # init value
                    match_feature = None
                    for t, feat in local_feature_list_source:
                        if abs(t - at_time / 1000) < min_diff:
                            min_diff = abs(t - at_time / 1000)
                            match_feature = feat
                    assert match_feature is not None
                    local_feature_list.append(match_feature)
            else:
                local_feature_list = [feat for _, feat in local_feature_list_source]
            local_feature_list = torch.stack(local_feature_list)
            assert video_res.shape[0] == local_feature_list.shape[0]

            return (video_res, local_feature_list, None), label_list
        except Exception as e:
            logger.warning("Failed to load template: %s, error: %s", self.tid_list[index], e)
            return self.__getitem__((index + 1) % self.__len__())


@DATASET_REGISTRY.register()
class TransitionClassificationDataset(Dataset):
    """
    this dataset is used for training transition embeddings
    """

    def __init__(self, cfg: CfgNode, mode: str):
        super(TransitionClassificationDataset, self).__init__()
        assert mode in ("train", "test", "val")
        cfg = cfg.DATASET.TRANSITION_CLASSIFICATION

        self.video_root = cfg.TEMPLATE_ROOT
        self.frame_per_clip = cfg.FRAME_PER_CLIP

        with open(cfg.JSON_ANNOTATION, "r") as f:
            json_annotation: dict = json.load(f)
        if mode in ["train", "test"]:  # test mode also use the train set, to generate embedding from the training set
            json_annotation["templates"] = json_annotation["templates"]["train"]
        else:
            json_annotation["templates"] = json_annotation["templates"]["test"]

        self.class_statistic = json_annotation["statistic"]["transition"]
        if cfg.WITH_DIRECT_CUT:
            self.class_dict = {t: i for i, t in enumerate(list(self.class_statistic)[:cfg.NUM_CLASSES])}
        else:
            self.class_dict = {t: i for i, t in enumerate(list(self.class_statistic)[1:cfg.NUM_CLASSES + 1])}
        logger.debug("Transition type used for training: %s", [k for k, _ in self.class_dict.items()])

        # transform
        if mode in ["train"]:
            self.transform = Compose([
                _n_h_w_c2n_c_h_w,
                Resize(min(cfg.SIZE)),
                RandomResizedCrop(cfg.SIZE, scale=(0.8, 1.2))
            ])
        else:
            self.transform = Compose([
                _n_h_w_c2n_c_h_w,
                Resize(min(cfg.SIZE)),
                CenterCrop(cfg.SIZE)
            ])

        tid_transition_pair = []
        for tid, effects in json_annotation["templates"].items():
            for transition in effects["transition"]:
                if transition["name"] in self.class_dict:  # remove other class
                    tid_transition_pair.append((tid, transition))

        # collect data
        self.tid = []
        self.transition = []
        for tid, transition in tid_transition_pair:
            self.tid.append(tid)
            fixed_transition = transition
            # fix `direct cut`
            if fixed_transition["duration"] < 50:
                fixed_transition["duration"] += 200
                fixed_transition["start"] -= 100
                fixed_transition["end"] += 100
            self.transition.append(fixed_transition)  # also fix the error at the same time

    def __len__(self):
        return len(self.tid)

    @staticmethod
    def _resample(source_list, target_num):
        list_len = len(source_list)
        step_size = (list_len - 1) / (target_num - 1)

        res = []
        for i in range(target_num):
            res.append(source_list[round(i * step_size)])
        assert len(res) == target_num
        return res

    def __getitem__(self, index):
        tid = self.tid[index]
        transition = self.transition[index]

        video_file_name = os.path.join(self.video_root, tid, f"{tid[-4:]}out_video.mp4")
        pts, _ = read_video_timestamps(video_file_name)

        # the start and end time for transition
        start = transition["start"] / 1000
        end = transition["end"] / 1000
        pts_during_transition = []
        for p in pts:
            if start < float(p) < end:
                pts_during_transition.append(p)

        # fault tolerance
        if len(pts_during_transition) == 0:
            logger.debug("No frame in template %s transition annotation: %s, video path: %s",
                         tid, transition, video_file_name)
            return self[index + 1]

        video_frame = [read_video_frame(video_file_name, p) for p in pts_during_transition]
        if len(video_frame) > self.frame_per_clip:
            video_frame = self._resample(video_frame, self.frame_per_clip)

        assert 0 < len(video_frame) <= self.frame_per_clip, f"Frame number is invalid: {len(video_frame)}"
        if len(video_frame) < self.frame_per_clip:
            video_frame = video_frame + [video_frame[-1], ] * (self.frame_per_clip - len(video_frame))
        video_frame = torch.stack(video_frame, dim=0)
        video_frame = self.transform(video_frame)

        label = self.class_dict[transition["name"]]

        return video_frame, label
