# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pytest
import requests
import subprocess
import sys
import time


class ServerRunner:
    def __init__(self, args):
        self.url = "http://localhost:8000"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "turbine_serving.llm.entrypoints.api_server",
            ]
            + args,
            env=env,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        self._wait_for_ready()

    def _wait_for_ready(self):
        start = time.time()
        while True:
            try:
                if requests.get(f"{self.url}/health").status_code == 200:
                    return
            except Exception as e:
                if self.process.poll() is not None:
                    raise RuntimeError("API server processs terminated") from e
            time.sleep(0.25)
            if time.time() - start > 30:
                raise RuntimeError("Timeout waiting for server start") from e

    def __del__(self):
        try:
            process = self.process
        except AttributeError:
            pass
        else:
            process.terminate()
            process.wait()


@pytest.fixture(scope="session")
def server():
    runner = ServerRunner([])
    yield runner


def test_basic(server: ServerRunner):
    ...
