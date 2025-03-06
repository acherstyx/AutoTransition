# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 13:57
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : download_videos.py

import argparse
import contextlib
import json
import logging
import os
import shutil
import time
import traceback

import joblib
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def download(url, out):
    r = requests.get(url, allow_redirects=True, timeout=30, stream=True)
    assert r.headers["Content-Type"] == 'video/mp4', f"Content-Type is {r.headers['Content-Type']}, {r.content}"
    open(out, "wb").write(r.content)


def download_template_video(output_dir, tid, url, verbose=False):
    for i in range(3):
        try:
            os.makedirs(os.path.join(output_dir, tid), exist_ok=True)
            download(url, os.path.join(output_dir, tid, f"{tid[-4:]}out_video.mp4.temp"))
        except Exception as e:
            if verbose:
                print(f"Download error: {tid}", traceback.format_exc())
            shutil.rmtree(os.path.join(output_dir, tid), ignore_errors=True)
            time.sleep(1)
            continue
        os.rename(os.path.join(output_dir, tid, f"{tid[-4:]}out_video.mp4.temp"),
                  os.path.join(output_dir, tid, f"{tid[-4:]}out_video.mp4"))
        return True
    logger.info(f"Failed to download video {tid}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-P", "--process", type=int, default=1)
    parser.add_argument("--aria2c_input_file", action="store_true")
    parser.add_argument("-v", "-V", "--verbose", action="store_true")

    args = parser.parse_args()

    with open(args.annotation, 'r') as f:
        annotation = json.load(f)

    tid_list = list(annotation["templates"]["train"].items()) + list(annotation["templates"]["test"].items())
    download_list = [(tid, info["url"]) for tid, info in tid_list]

    if not args.aria2c_input_file:
        with tqdm_joblib(tqdm(total=len(download_list), desc="Downloading")):
            result = joblib.Parallel(n_jobs=args.process)(
                joblib.delayed(download_template_video)(args.output_dir, tid, url, args.verbose) for tid, url in
                download_list
            )
        print(f"Success: {sum(result)}")
        print(f"Failed: {len(result) - sum(result)}")
    else:
        with open("aria2c_input_file.txt", "w") as f:
            for tid, url in download_list:
                f.write(f"{url}\n")
                f.write(f"\tout={os.path.join(args.output_dir, tid, f'{tid[-4:]}out_video.mp4')}\n")


if __name__ == '__main__':
    main()
