# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence

import argparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import sys
import uvicorn

app = FastAPI()


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


def main(clargs: Sequence[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="Root path to use for installing behind path based proxy.",
    )
    parser.add_argument(
        "--timeout-keep-alive", type=int, default=5, help="Keep alive timeout"
    )
    args = parser.parse_args(clargs)

    app.root_path = args.root_path
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
