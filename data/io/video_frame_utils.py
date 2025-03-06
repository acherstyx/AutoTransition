# -*- coding: utf-8 -*-
# @Time    : 2021/11/11 6:57 PM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : video_frame_utils.py

import logging
import os
import shutil
import traceback
from fractions import Fraction
from typing import *

import einops
import torch
import torchvision
import torchvision.transforms.functional as F
import tqdm
from torch.utils import data

logger = logging.getLogger(__name__)

__all__ = ["convert_tasker", "clean_tasker", "read_video", "read_video_timestamps", "read_video_with_step",
           "glob_video_files", "count_video_frames", "compute_pts_tasker", "check_valid_tasker", "video2rgb_frame",
           "read_video_frame"]

"""
contain API functions similar to torchvision.io, but for videos which are stored in the form of *.jpg files.
see torchvision.io: https://pytorch.org/vision/stable/io.html
"""


def video2rgb_frame(video_path: str,
                    overwrite: bool = False,
                    delete_video: bool = False,
                    resize_size: Union[int, List] = 320,
                    postfix: str = ".frame", ):
    """
    convert video to rgb frames, save to folder: {video_path}{postfix}
    the image is named by pts (unit=sec)
    :param video_path: a single video
    :param postfix: save images into a folder, the path is set to `video_path+postfix`
    :param overwrite: bool
    :param delete_video: if delete video after converting
    :param resize_size:
    :return:
    """
    # if postfix == "", the video file will be replaced with a folder containing all video frames,
    # and the video should be deleted to create a folder with the same name
    assert postfix != "" or delete_video, "delete the video or specify another postfix"
    assert os.path.exists(video_path), f"file not found: {video_path}"
    assert os.path.isfile(video_path), f"is a folder, not file: {video_path}"

    folder = video_path + postfix

    # check existing folder
    if os.path.exists(folder):
        if overwrite:
            shutil.rmtree(folder)
        else:
            return
    os.mkdir(folder)

    # load using torchvision API
    video, _, fps = torchvision.io.read_video(video_path, pts_unit="sec")

    for cur_index, frame in enumerate(video):
        cur_pts = Fraction(cur_index, round(fps["video_fps"]))  # calculate pts from frame rate and frame index
        frame = einops.rearrange(frame, "h w c -> c h w")
        frame = F.resize(frame, resize_size)
        torchvision.io.write_jpeg(frame,
                                  os.path.join(folder, str(cur_pts).replace("/", "_") + ".jpg"))

    if delete_video:
        os.remove(video_path)


def _clean_frame(video_path: str, postfix: str):
    folder = video_path + postfix
    shutil.rmtree(folder, ignore_errors=True)


class _Converter(data.Dataset):
    def __init__(self, video_list, postfix: str, overwrite: bool, delete_video: bool, resize_size: int):
        super(_Converter, self).__init__()
        self.video_list = video_list
        self._convert_func = lambda x: video2rgb_frame(x,  # video
                                                       postfix=postfix,
                                                       overwrite=overwrite,
                                                       delete_video=delete_video,
                                                       resize_size=resize_size)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        try:
            self._convert_func(self.video_list[index])
        except Exception as e:
            logger.warning(f"Failed to convert video: {self.video_list[index]}, error: {e}. "
                           f"Error stack for debugging: {traceback.format_exc()}")


def convert_tasker(convert_task_list, postfix=".frame", overwrite=False, delete_video=False, resize_size=320,
                   num_workers=None):
    """
    convert all videos in the convert_task_list to frame and store them in the folder for each video
    :param convert_task_list:
    :param postfix: video_path + postfix => folder
    :param overwrite: remove the folder if already exists
    :param delete_video: after converting, delete the video
    :param resize_size: resize image before saving to disk
    :param num_workers: worker
    """
    convert_dl = data.DataLoader(_Converter(convert_task_list,
                                            postfix=postfix,
                                            overwrite=overwrite,
                                            delete_video=delete_video,
                                            resize_size=resize_size),
                                 batch_size=1,
                                 shuffle=False,
                                 prefetch_factor=10,
                                 num_workers=os.cpu_count() if num_workers is None else num_workers,
                                 collate_fn=lambda x: x)
    for _ in tqdm.tqdm(convert_dl, desc="Converting Videos"):
        pass


def clean_tasker(clean_task_list: list, postfix: str = ".frame"):
    """
    opposite of convert_tasker, remove all folders
    :param clean_task_list: same as convert_task_list as in convert_tasker()
    :param postfix: video_path + postfix => folder
    """
    for v in tqdm.tqdm(clean_task_list, desc="Cleaning Videos"):
        _clean_frame(v, postfix)


class _VideoTimestampsDataset(data.Dataset):
    def __init__(self, _video_paths, use_frame):
        self.video_paths = _video_paths
        self.use_frame = use_frame

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        if self.use_frame:
            return read_video_timestamps(self.video_paths[idx], pts_unit="sec")
        else:
            return torchvision.io.read_video_timestamps(self.video_paths[idx], pts_unit="sec")


def _dummy_collect_fn(x):
    return x


def compute_pts_tasker(compute_video_list: object,
                       enable_process_bar: object = False,
                       use_frame: object = False,
                       desc: object = "Computing pts") -> Tuple[List[List[Fraction]], List[float]]:
    pts_data_loader = data.DataLoader(_VideoTimestampsDataset(compute_video_list, use_frame=use_frame),
                                      batch_size=1,
                                      prefetch_factor=10,
                                      num_workers=os.cpu_count(),
                                      collate_fn=_dummy_collect_fn,
                                      multiprocessing_context="fork")
    video_pts = []
    video_fps = []
    if enable_process_bar:
        pts_data_loader = tqdm.tqdm(pts_data_loader, desc=desc)
    for video_pts_batch in pts_data_loader:
        clips, fps = list(zip(*video_pts_batch))
        video_pts.extend(clips)
        video_fps.extend(fps)
    assert len(video_pts) == len(compute_video_list)
    del pts_data_loader
    return video_pts, video_fps


class _VideoCheckDataset(data.Dataset):
    def __init__(self, video_list: Union[List[str], List[bytes]]):
        self.video_list = video_list

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        video = self.video_list[index]
        v_frame, _, _ = torchvision.io.read_video(video, start_pts=Fraction(0), end_pts=Fraction(0), pts_unit="sec")
        if v_frame.shape[0] > 0:
            return True
        else:
            return False


def check_valid_tasker(check_video_list, enable_process_bar=False, use_frame=False):
    assert use_frame is False, "Check videos using frame is not supported! set it to False."

    check_ds = _VideoCheckDataset(check_video_list)
    check_dl = data.DataLoader(check_ds,
                               num_workers=os.cpu_count(),
                               collate_fn=_dummy_collect_fn,
                               shuffle=False,
                               multiprocessing_context="fork")
    if enable_process_bar:
        check_dl = tqdm.tqdm(check_dl, desc="Checking videos")
    res = []
    for is_valid in check_dl:
        res.extend(is_valid)
    assert len(res) == len(check_video_list)
    return res


def read_video_frame(filename: str,
                     pts: Fraction,
                     postfix: str = ".frame"):
    frame_dir = str(filename) + postfix

    img_name = str(pts).replace("/", "_") + ".jpg"
    assert os.path.exists(os.path.join(frame_dir, img_name))
    cur_img = torchvision.io.read_image(os.path.join(frame_dir, img_name))
    res = einops.rearrange(cur_img, "c h w -> h w c")
    assert isinstance(res, torch.Tensor)
    return res


def read_video_with_step(filename: str,
                         start_pts: Fraction = None,
                         end_pts: Fraction = None,
                         step: int = None,
                         postfix: str = ".frame",
                         _precomputed_pts: List[Fraction] = None) -> torch.Tensor:
    """
    read video from corresponding image folder, with step
    :param filename:
    :param start_pts:
    :param end_pts:
    :param step:
    :param postfix:
    :param _precomputed_pts: tensor of video with the shape of (N,H,W,C)
    :return:
    """
    if _precomputed_pts is not None:
        pts = _precomputed_pts
    else:
        pts, fps = read_video_timestamps(filename)
    frame_dir = str(filename) + postfix

    if start_pts is not None:
        start_index = pts.index(start_pts)
    else:
        start_index = 0
    if end_pts is not None:
        end_index = pts.index(end_pts)
    else:
        end_index = -2

    res = []
    for cur_pts in pts[start_index: end_index + 1:step]:
        img_name = str(cur_pts).replace("/", "_") + ".jpg"
        cur_img = torchvision.io.read_image(os.path.join(frame_dir, img_name))
        res.append(cur_img)

    if res:
        res = torch.stack(res)
        res = einops.rearrange(res, "n c h w -> n h w c")
    else:
        logger.debug(f"no frame in folder: {frame_dir}")
        res = torch.Tensor()

    return res


def read_video(filename: str,
               start_pts: Fraction = None,
               end_pts: Fraction = None,
               postfix: str = ".frame",
               _precomputed_pts: List[Fraction] = None) -> torch.Tensor:
    """
    read video from corresponding image folder
    :param filename:
    :param start_pts:
    :param end_pts:
    :param postfix:
    :param _precomputed_pts:
    :return: tensor of video with the shape of (N,H,W,C)
    """
    return read_video_with_step(filename=filename,
                                start_pts=start_pts,
                                end_pts=end_pts,
                                step=1,  # with step=1
                                postfix=postfix,
                                _precomputed_pts=_precomputed_pts)


def read_video_timestamps(filename, postfix: str = ".frame", pts_unit="sec") -> Tuple[List[Fraction], float]:
    """
    compute video pts based on image frames stored in the image folder
    :param filename: video file name
    :param postfix:
    :param pts_unit: only support pts_unit="sec"
    :return: pts: list of Fraction, fps: fps if frame number >=2, else None
    """
    assert pts_unit == "sec", "only support pts_unit 'sec'"
    frame_dir = str(filename) + postfix

    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"video frame directory not exists: {frame_dir}")

    frame_files = os.listdir(frame_dir)  # original files
    frames = [f.split(".")[0] for f in frame_files if f.endswith(".jpg")]  # filter

    pts = []
    for cur_frame in frames:
        cur_frame = [int(f) for f in cur_frame.split("_")]
        if len(cur_frame) == 2:
            pts.append(Fraction(*cur_frame))
        elif len(cur_frame) == 1:
            pts.append(Fraction(cur_frame[0], 1))
        else:
            raise RuntimeError(f"file name is invalid: {cur_frame}")

    pts = sorted(pts)
    if len(pts) < 2:
        fps = None
    else:
        fps = float(pts[1].denominator)
    return pts, fps


def count_video_frames(filename, postfix: str = ".frame") -> int:
    frame_dir = str(filename) + postfix

    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"video frame directory not exists: {frame_dir}")

    frame_files = os.listdir(frame_dir)  # original files
    frames = [f for f in frame_files if f.endswith(".jpg")]  # filter
    return len(frames)


def glob_video_files(root_dir: str,
                     extension: Union[List[str], Tuple[str]] = ("mp4", "avi"),
                     use_frame_folder: bool = False,
                     postfix: str = ".frame"):
    """
    list all the videos in a given folder (recursive), or frame folders with given postfix
    :param root_dir:
    :param extension:
    :param use_frame_folder: if True, ignore videos and search frame folders instead, the postfix will be remove from the path of folder
    :param postfix:
    :return:
    """
    assert os.path.isdir(root_dir), f"{root_dir} is not a directory"

    def check_frame_folder(cur_dir):
        return True if cur_dir.endswith(postfix) else False

    def check_extension(filename):
        return True if any(filename.endswith(e) for e in extension) else False

    res = []
    for root, dirs, files in tqdm.tqdm(os.walk(root_dir)):
        if use_frame_folder:
            for d in dirs:
                if check_frame_folder(d):
                    res.append(os.path.join(root, d)[:-len(postfix)])
        else:
            for f in files:
                if check_extension(f):
                    res.append(os.path.join(root, f))
    return res
