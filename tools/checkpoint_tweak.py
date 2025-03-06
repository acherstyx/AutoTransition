# -*- coding: utf-8 -*-
# @Time    : 2021/11/30 2:42 PM
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : checkpoint_tweak.py

import argparse
from collections import OrderedDict
from typing import *

import torch

parser = argparse.ArgumentParser(description="Checkpoint tool")
parser_sub = parser.add_subparsers(required=True, dest="function")

# different function
parser_ls = parser_sub.add_parser("ls")
parser_ls.add_argument("checkpoint", help="path to checkpoint file", type=str)
parser_ls.add_argument("--key", "-k", help="if the checkpoint is saved as python dict, give the key value",
                       default=None, type=str)
parser_ls.add_argument("-v", help="verbose", default=False, action="store_true")

parser_replace = parser_sub.add_parser("replace")
parser_replace.add_argument("checkpoint", help="path to checkpoint file", type=str)
parser_replace.add_argument("--key", "-k", help="if the checkpoint is saved as python dict, give the key value",
                            default=None, type=str)
parser_replace.add_argument("old_key", type=str)
parser_replace.add_argument("new_key", type=str)
parser_replace.add_argument("-r", "--remove", default=False, action="store_true",
                            help="Remove parameters which key value is not start with old_key")
parser_replace.add_argument("--save", type=str, help="save new checkpoint to a ckpt file", default=None)


def ls(args):
    checkpoint = torch.load(args.checkpoint)
    if args.key is not None:
        checkpoint = checkpoint[args.key]

    keys = [k for k, v in checkpoint.items()]
    for k in keys:
        k: str
        if args.v:
            print(k)
        else:
            print(".".join(k.split(".")[:-1]))


def replace(args):
    checkpoint = torch.load(args.checkpoint)
    if args.key is not None:
        checkpoint = checkpoint[args.key]

    if args.remove:
        checkpoint = {k: v for k, v in checkpoint.items() if k.startswith(args.old_key)}
    checkpoint: Dict[str]
    checkpoint = {k.replace(args.old_key, args.new_key): v for k, v in checkpoint.items()}
    checkpoint = {k[1:] if k.startswith(".") else k: v for k, v in checkpoint.items()}
    checkpoint = OrderedDict(checkpoint)

    print("Result")
    for key in checkpoint.keys():
        print(key)

    if args.save is not None:
        torch.save(checkpoint, args.save)


if __name__ == '__main__':
    my_args = parser.parse_args()

    exec(f"{my_args.function}(my_args)")
