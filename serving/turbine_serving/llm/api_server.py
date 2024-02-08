# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Heavily adapted from the vllm api_server.py.

from typing import AsyncGenerator, Optional, Sequence

import argparse
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import sys
import uuid
import uvicorn

from .logging import get_logger

from .service import (
    create_mock_generate_service,
    GenerateService,
    GenerateRequest,
)

logger = get_logger("turbine_serving.llm.api_server")
app = FastAPI()
service: Optional[GenerateService] = None


def get_service() -> GenerateService:
    assert service is not None, "Service was not initialized"
    return service


@app.get("/health")
async def health() -> Response:
    get_service()
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    service = get_service()
    r = await request.json()
    prompt = r.pop("prompt")
    stream = bool(r.pop("stream", False))
    request_id = uuid.uuid4().hex

    generate_request = GenerateRequest(request_id=request_id, prompt=prompt)
    result_parts = service.handle_request(generate_request)

    if stream:
        # TODO: This isn't entirely matching how others do it: we should be returning
        # the full result on each update.
        async def stream_contents() -> AsyncGenerator[bytes, None]:
            async for part in result_parts:
                response_record = json.dumps({"text": part.text})
                yield (response_record + "\0").encode()

        return StreamingResponse(stream_contents())

    # Non-streaming just reads to the final.
    async for result_part in result_parts:
        if await request.is_disconnected():
            # Abort.
            await service.abort(generate_request.request_id)
            return Response(status_code=499)

    assert result_part is not None, "No results generated!"
    return JSONResponse({"text": result_part.text})


def main(clargs: Sequence[str]):
    global service
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
    parser.add_argument(
        "--testing-mock-service",
        action="store_true",
        help="Enable the mock testing service",
    )
    parser.add_argument(
        "--device-uri", type=str, default="local-task", help="Device URI to serve on"
    )

    args = parser.parse_args(clargs)

    # Spin up the device machinery.
    # Note that in the future, for multi-device, we will need more scaffolding for
    # configuration and bringup, obviously.
    from .session import DeviceSession

    device_session = DeviceSession(uri=args.device_uri)

    if args.testing_mock_service:
        logger.info("Enabling mock LLM generate service")
        service = create_mock_generate_service()

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
