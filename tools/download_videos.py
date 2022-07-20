# -*- coding: utf-8 -*-
# @Time    : 2022/7/14 13:57
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : download_videos.py

import os
import json
import argparse
import requests
import time
from utils.multithread import MultithreadTasker
import shutil
import logging

logger = logging.getLogger(__name__)


def download(url, out):
    r = requests.get(url, allow_redirects=True, timeout=30)
    assert r.headers["Content-Type"] == 'video/mp4'
    open(out, "wb").write(r.content)


def download_template_video(output_dir, tid, url):
    for i in range(3):
        try:
            os.makedirs(os.path.join(output_dir, tid), exist_ok=True)

            download(url, os.path.join(output_dir, tid, f"{tid[-4:]}out_video.mp4.temp"))
        except Exception as e:
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

    args = parser.parse_args()

    with open(args.annotation, 'r') as f:
        annotation = json.load(f)

    tid_list = list(annotation["templates"]["train"].items()) + list(annotation["templates"]["test"].items())
    download_list = [(tid, info["url"]) for tid, info in tid_list]

    download_tasker = MultithreadTasker(download_template_video, cpu_num=args.process, prefetch_factor=1024,
                                        process_bar=True, process_bar_desc="Downloading")
    for tid, url in download_list:
        download_tasker.add_task([args.output_dir, tid, url])
    result = download_tasker.run()

    print(f"Success: {sum(result)}")
    print(f"Failed: {len(result) - sum(result)}")


if __name__ == '__main__':
    main()
