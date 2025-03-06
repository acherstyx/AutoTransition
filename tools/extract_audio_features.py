# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 16:33
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : extract_audio_features.py

import argparse
import json
import os

import torch
from tqdm import tqdm

from autotransition.preprocess.harmonic.extract import MusicLocalFeatureSynchronized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=str, help="template dir")
    parser.add_argument("annotation", type=str, help="annotation file")
    parser.add_argument("--model_path", type=str, help="path to pretrained model weights",
                        default="./harmonic_multisim_multisim_euc_b128_l8_ccenterwin_dsync_epoch_14.pth")
    parser.add_argument("--num_worker", type=int, default=16)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    extract_audio_feature_tasker = MusicLocalFeatureSynchronized(model_path=args.model_path,
                                                                 device=torch.device(0) if args.cuda else
                                                                 torch.device('cpu'))

    with open(args.annotation, "r") as f:
        annotation = json.load(f)
    template_annotation = {}
    template_annotation.update(annotation["templates"]["train"])
    template_annotation.update(annotation["templates"]["test"])

    tid_list = [tid for tid in os.listdir(args.template) if len(tid) == 19 and all(c in "1234567890" for c in tid)]

    extract_tid_list = []
    for tid in tqdm(tid_list, desc="Building task"):
        if tid in template_annotation:
            extract_audio_feature_tasker.append_task(os.path.join(args.template, tid, f"{tid[-4:]}out_video.mp4"),
                                                     template_annotation[tid]["transition"])
            extract_tid_list.append(tid)
    for tid, audio_feature in zip(tqdm(extract_tid_list, desc="Local features"),
                                  iter(extract_audio_feature_tasker.run(args.num_worker))):
        if audio_feature is None:
            continue
        assert os.path.join(args.template, tid, f"{tid[-4:]}out_video.mp4") == audio_feature["metadata"]["audio_file"]
        with open(os.path.join(args.template, tid, "audio_sync_local_feature.json"), "w") as f:
            json.dump(audio_feature, f)


if __name__ == '__main__':
    main()
