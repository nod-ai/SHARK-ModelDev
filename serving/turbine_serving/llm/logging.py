# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import sys


# Whether debug assertions are disabled.
NDEBUG: bool = False

_default_log_level = os.getenv("TURBINE_LOG_LEVEL", "DEBUG")


class DefaultFormatter(logging.Formatter):
    def __init__(self):
        super().__init__(
            "%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s",
            "%m-%d %H:%M:%S",
        )


def _setup_logger():
    root_logger = logging.getLogger("turbine_serving")
    root_logger.setLevel(logging.DEBUG)
    default_handler = logging.StreamHandler(sys.stderr)
    default_handler.flush = sys.stderr.flush
    default_handler.setLevel(_default_log_level)
    default_handler.setFormatter(DefaultFormatter())
    root_logger.addHandler(default_handler)
    root_logger.propagate = False
    return root_logger, default_handler


root_logger, default_handler = _setup_logger()


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(_default_log_level)
    logger.addHandler(default_handler)
    logger.propagate = False
    return logger
