# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from iree.runtime import (  # type: ignore
    HalElementType,
)

from turbine_serving.llm.session import DeviceSession
from turbine_serving.llm.config import (
    CacheParams,
    ModelParams,
    ServiceParams,
)

from turbine_serving.llm.service import (
    BatchGenerateRequest,
    GenerateRequest,
    GenerateResponsePart,
)

from turbine_serving.llm.impl.service_v1 import (
    GenerateServiceV1,
)

from turbine_serving.llm.testing.fake_v1_module import (
    create_fake_module,
)


@pytest.fixture
def model_params() -> ModelParams:
    return ModelParams(
        module_name="AwesomeLLM",
        module_abi_version=1,
        attn_dtype=HalElementType.FLOAT_16,
        max_seq_len=128,
        transformer_block_count=32,
        attn_head_count=32,
        attn_head_dim=128,
        prefill_batch_sizes=[1, 4, 16],
        decode_batch_sizes=[1, 4, 16],
    )


@pytest.fixture
def session(model_params: ModelParams, scope="session"):
    from iree.runtime._binding import disable_leak_checker  # type: ignore

    disable_leak_checker()
    session = DeviceSession(uri="local-task")
    lms = session.create_module_set("AwesomeLLM", context_count=1)
    lms.add(
        create_fake_module("AwesomeLLM", model_params=model_params),
    )
    lms.initialize()
    yield session
    session.shutdown()
    del session


@pytest.fixture
def cache_params(model_params: ModelParams) -> CacheParams:
    return CacheParams(model=model_params, device_block_count=128, block_pos_stride=16)


@pytest.fixture
def service(
    session: DeviceSession, cache_params: CacheParams, model_params: ModelParams
):
    params = ServiceParams(cache=cache_params, model=model_params)
    return GenerateServiceV1(session, params)


def test_single(service: GenerateServiceV1):
    def callback(response: list[GenerateResponsePart]):
        print("RESPONSE:", response)

    request = BatchGenerateRequest(
        requests=[
            GenerateRequest(
                "1",
                "hello, tell me a story",
                [3, 4, 5, 12, 23, 88, 10, 2, 5, 9, 12, 13, 99, 56, 33, 124, 73],
            ),
            GenerateRequest("2", "goodbye", [9, 10]),
        ],
    )
    service.start_prefill(request)
