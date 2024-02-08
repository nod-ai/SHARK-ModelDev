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
                "turbine_serving.llm.api_server",
                "--testing-mock-service",
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
            time.sleep(1.0)
            if time.time() - start > 30:
                raise RuntimeError("Timeout waiting for server start")

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


def test_health(server: ServerRunner):
    # Health check is part of getting the fixture.
    ...


def test_generate_non_streaming(server: ServerRunner):
    resp = requests.post(
        f"{server.url}/generate",
        json={
            "prompt": "Hi Bob",
        },
    )
    resp.raise_for_status()
    d = resp.json()
    assert d["text"] == "Hi Bob", repr(d)


def test_generate_streaming(server: ServerRunner):
    resp = requests.post(
        f"{server.url}/generate", json={"prompt": "Hi Bob!", "stream": True}
    )
    resp.raise_for_status()
    full_contents = resp.content
    expected_contents = b'{"text": "Hi Bob!"}\x00' * 5
    assert (
        full_contents == expected_contents
    ), f"Expected {expected_contents!r} vs {full_contents!r}"
