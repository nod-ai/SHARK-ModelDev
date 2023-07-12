# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
from typing import List, Optional, Sequence, Union

from ..support.compiler_api import Compiler, Pipeline

import iree.runtime as rt

DEFAULT_COMPILER_FLAGS = (
    # Enable asynchronous calling convention.
    "--iree-execution-model=async-external",
)


@functools.lru_cache(maxsize=None)
def get_vm_instance() -> rt.VmInstance:
    return rt.VmInstance()


class DeviceState:
    def __init__(
        self, *, driver: Union[str, rt.HalDriver], device: Optional[rt.HalDevice] = None
    ):
        self.instance = get_vm_instance()
        self.driver = (
            driver if isinstance(driver, rt.HalDriver) else rt.get_driver(driver)
        )
        self.device = device if device else self.driver.create_default_device()


class InputModule:
    """Input to the compiler.

    This can carry additional metadata about the module which controls how
    it is executed.
    """

    def __init__(self, module_bytecode: bytes):
        self.module_bytecode = module_bytecode

    def __repr__(self):
        return self.module_bytecode.decode()


class SpecializedExecutable:
    """A concrete executable that has been specialized in some way."""

    def __init__(
        self,
        user_module: rt.VmModule,
        device_state: DeviceState,
        entry_name: str = "forward",
    ):
        self.user_module = user_module
        self.vm_context = rt.VmContext(
            device_state.instance,
            (
                rt.create_hal_module(device_state.instance, device_state.device),
                user_module,
            ),
        )
        self.device_state = device_state
        self.entry_function = self.user_module.lookup_function(entry_name)

    def __call__(self, *inputs):
        raise NotImplementedError()


class JittableExecutable:
    """A compilable executable which can be run with the IREE runtime.

    Note that this is modeled as a second-level JIT beyond what Dynamo itself
    caches. This allows further specialization.

    For right now, we are not actually further specializing. We are also
    compiling synchronously because it is easy to debug. We will lift both
    restrictions at a later date.
    """

    def __init__(
        self,
        input_module: InputModule,
        device_state: DeviceState,
        *,
        compiler_flags: Sequence[str]
    ):
        self.device_state = device_state
        self.compiler_flags = DEFAULT_COMPILER_FLAGS + tuple(compiler_flags)
        self.input_module = input_module
        self._default_spec = self._compile_default_spec()

    def __call__(self, *inputs):
        print("Inputs:", inputs)
        return self._default_spec(*inputs)

    def _compile_default_spec(self) -> SpecializedExecutable:
        # TODO: Should be caching compiler instances.
        compiler = Compiler()
        compiler.set_flags(*self.compiler_flags)
        pipeline = compiler.load_buffer(
            self.input_module.module_bytecode, buffer_name="dynamo"
        )
        pipeline.execute()

        # Output to runtime.
        vmfb_output = compiler.open_output_membuffer()
        pipeline.output_vm_bytecode(vmfb_output)
        vmfb_module = rt.VmModule.wrap_buffer(
            self.device_state.instance,
            vmfb_output.map_memory(),
            destroy_callback=vmfb_output.close,
        )
        return SpecializedExecutable(vmfb_module, self.device_state)
