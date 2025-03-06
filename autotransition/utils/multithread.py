# -*- coding: utf-8 -*-
# @Time    : 2021/12/10 9:22 PM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : multithread.py

import os
from typing import *

import tqdm
from torch.utils import data

"""
Enabling multithread processing through torch DataSet and DataLoader
"""


class _ThreadDataset(data.Dataset):

    def __init__(self, args_list: List[Any], process_func: Callable):
        self.args_list = args_list
        self.callable_func = process_func

    def __len__(self):
        return len(self.args_list)

    def __getitem__(self, index):
        return self.callable_func(*self.args_list[index])


def _dummy_collate_fn(x):
    return x


def multithread_tasker(args_list: List,
                       process_func: Callable,
                       cpu_num=os.cpu_count(),
                       prefetch_factor=2,
                       process_bar=False,
                       process_bar_desc=None):
    ds = _ThreadDataset(args_list, process_func)
    dl = data.DataLoader(ds,
                         batch_size=1,
                         collate_fn=_dummy_collate_fn,
                         num_workers=cpu_num,
                         prefetch_factor=prefetch_factor)

    if process_bar:
        dl = tqdm.tqdm(dl, desc=process_bar_desc)

    ret = []
    for processing_output in dl:
        ret.extend(processing_output)

    return ret


class MultithreadTasker:
    def __init__(self, process_func: Callable,
                 cpu_num=os.cpu_count(),
                 prefetch_factor=2,
                 process_bar=False,
                 process_bar_desc=None):
        self.func = process_func
        self.args_list = []
        self.cpu_num = cpu_num
        self.process_bar = process_bar
        self.process_bar_desc = process_bar_desc
        self.prefetch_factor = prefetch_factor

    def __len__(self):
        return len(self.args_list)

    def add_task(self, args: List):
        self.args_list.append(args)

    def run(self, clear=True):
        ret = multithread_tasker(self.args_list,
                                 process_func=self.func,
                                 cpu_num=self.cpu_num,
                                 prefetch_factor=self.prefetch_factor,
                                 process_bar=self.process_bar,
                                 process_bar_desc=self.process_bar_desc)
        if clear:
            self.clear()

        return ret

    def clear(self):
        self.args_list = []
