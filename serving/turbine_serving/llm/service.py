# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import AsyncIterator, Callable, Optional

from abc import abstractmethod, ABC
import asyncio
from dataclasses import dataclass

from .session import (
    HostContext,
)

########################################################################################
# User-level single request service
########################################################################################


@dataclass
class GenerateRequest:
    """Encapsulates a request to perform LLM generation.

    Requests are bootstrapped from user values and then pumped through the pipeline,
    receiving additional elaboration needed to actually begin generation.
    """

    # Client set fields
    request_id: str
    prompt: str

    # Fields that are set as the request is processed.
    prompt_token_ids: Optional[list[int]] = None

    @property
    def required_prompt_token_ids(self) -> list[int]:
        ids = self.prompt_token_ids
        assert ids is not None
        return ids


@dataclass
class GenerateResponsePart:
    """A response part from an LLM generation request."""

    request: GenerateRequest
    index: int
    token_ids: list[int]

    # Fields that can be set as the response is post-processed.
    text: Optional[str] = None
    finished: bool = False


class GenerateService(ABC):
    """Asynchronous generator service which processes requests into response parts."""

    @abstractmethod
    def handle_request(
        self,
        request: GenerateRequest,
    ) -> AsyncIterator[GenerateResponsePart]:
        """Generates response parts for a request."""
        ...

    @abstractmethod
    async def abort(self, request_id: str) -> None:
        """Aborts a submitted request."""
        ...


########################################################################################
# Batch generation service
# This service is completely asynchronous and operates on a BatchGenerateRequest as
# a state machine. It is expected to have an external actor stepping it through
# states.
########################################################################################


class BatchGenerateService(ABC):
    """Handles generation of a batch of requests."""

    __slots__ = []  # type: ignore

    # def start_prefill(self, request: BatchGenerateRequest):
    #     ...
    @abstractmethod
    def start(self) -> "BatchGenerateState":
        ...


class BatchGenerateState(ABC):
    """In-progress batch generation state."""

    __slots__ = [
        "host_context",
    ]

    def __init__(self, host_context: HostContext):
        self.host_context = host_context


########################################################################################
# Utilities
########################################################################################


class SyncGenerateFilter(GenerateService):
    """GenerateService filter which can synchronously pre/post process."""

    __slots__ = ["_next"]

    def __init__(self, next: GenerateService):
        self._next = next

    def filter_request(self, request: GenerateRequest):
        ...

    def filter_response(self, part: GenerateResponsePart):
        ...

    async def handle_request(
        self,
        request: GenerateRequest,
    ) -> AsyncIterator[GenerateResponsePart]:
        self.filter_request(request)
        async for part in self._next.handle_request(request):
            self.filter_response(part)
            yield part

    async def abort(self, request_id: str) -> None:
        """Aborts a submitted request."""
        await self._next.abort(request_id)


########################################################################################
# Testing and mock types
########################################################################################


def create_mock_generate_service() -> GenerateService:
    return DummyTokenizerService(EchoGenerateService())


class DummyTokenizerService(SyncGenerateFilter):
    """Tokenizer service which will map to code points.

    Useful for testing.
    """

    def filter_request(self, request: GenerateRequest):
        if request.prompt_token_ids is None:
            request.prompt_token_ids = [ord(c) for c in request.prompt]

    def filter_response(self, part: GenerateResponsePart):
        if part.text is None:
            part.text = "".join([chr(x) for x in part.token_ids])


class EchoGenerateService(GenerateService):
    """Dummy implementation of a generate service.

    It just echoes back the request five times after a delay.
    """

    def __init__(self, delay: float = 0.1):
        self._delay = delay

    async def handle_request(
        self,
        request: GenerateRequest,
    ) -> AsyncIterator[GenerateResponsePart]:
        next = None
        for i in range(5):
            if next:
                yield next
            assert request.prompt_token_ids, "Request lacks prompt tokens"
            next = GenerateResponsePart(
                request, i, request.prompt_token_ids, finished=False
            )
            await asyncio.sleep(self._delay)
        if next:
            next.finished = True
            yield next

    async def abort(self, request_id: str) -> None:
        pass
