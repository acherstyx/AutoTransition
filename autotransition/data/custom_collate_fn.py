# -*- coding: utf-8 -*-
# @Time    : 2021/12/28 6:16 PM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : custom_collate_fn.py

import torch
from torch.nn.utils import rnn

from .build import COLLATE_FN_REGISTER


@COLLATE_FN_REGISTER.register()
def visual_music_collate_fn(batch_data):
    batch_video = []
    batch_audio_feature = []
    batch_labels = []
    for (video_res, local_feature, global_feature), label_list in batch_data:
        audio_feature = local_feature
        batch_labels.append(label_list)
        batch_video.append(video_res)
        batch_audio_feature.append(audio_feature)
    batch_labels = rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-1)
    batch_label_mask = batch_labels != -1
    batch_labels[batch_labels == -1] = 0
    batch_video = rnn.pad_sequence(batch_video, batch_first=True, padding_value=0.0)
    batch_audio_mask = [torch.ones(music_feature.shape[0], dtype=torch.bool) for music_feature in batch_audio_feature]
    batch_audio_mask = rnn.pad_sequence(batch_audio_mask, batch_first=True)
    batch_audio_feature = rnn.pad_sequence(batch_audio_feature, batch_first=True, padding_value=0.0)

    return (batch_video, batch_audio_feature, batch_label_mask, batch_audio_mask), batch_labels
