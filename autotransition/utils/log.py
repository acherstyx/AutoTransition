# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 16:21
# @Author  : Yaojie Shen
# @Project : AutoTransition
# @File    : log.py

import logging
import os

level_dict = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "notset": logging.NOTSET
}


def setup_logging(cfg):
    # log file
    if len(str(cfg.LOG.LOG_FILE).split(".")) == 2:
        file_name, extension = str(cfg.LOG.LOG_FILE).split(".")
        log_file_debug = os.path.join(cfg.LOG.DIR, f"{file_name}_debug.{extension}")
        log_file_info = os.path.join(cfg.LOG.DIR, f"{file_name}_info.{extension}")
    elif len(str(cfg.LOG.LOG_FILE).split(".")) == 1:
        file_name = cfg.LOG.LOG_FILE
        log_file_debug = os.path.join(cfg.LOG.DIR, f"{file_name}_debug")
        log_file_info = os.path.join(cfg.LOG.DIR, f"{file_name}_info")
    else:
        raise ValueError("cfg.LOG.LOG_FILE is invalid: %s", cfg.LOG.LOG_FILE)

    console_level = cfg.LOG.LOG_CONSOLE_LEVEL
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter_debug = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    formatter_info = logging.Formatter(
        "[%(asctime)s][%(levelname)s]: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    root.handlers.clear()

    # log file
    if os.path.dirname(log_file_debug):  # dir name is not empty
        os.makedirs(os.path.dirname(log_file_debug), exist_ok=True)
    # console
    handler_console = logging.StreamHandler()
    handler_console.setLevel(level_dict[console_level.lower()])
    handler_console.setFormatter(formatter_info)
    root.addHandler(handler_console)
    # debug level
    handler_debug = logging.FileHandler(log_file_debug, mode="a")
    handler_debug.setLevel(logging.DEBUG)
    handler_debug.setFormatter(formatter_debug)
    root.addHandler(handler_debug)
    # info level
    handler_info = logging.FileHandler(log_file_info, mode="a")
    handler_info.setLevel(logging.INFO)
    handler_info.setFormatter(formatter_info)
    root.addHandler(handler_info)

    root.propagate = False
