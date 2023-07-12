# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools

#from ..executor import DeviceState, InputModule, JittableExecutable
#from ..script_importer import ScriptImporter, make_simple_dynamo_backend

import torch


class GraphLowering(torch.fx.Interpreter):
    def call_function(self, target, *args, **kwargs):
        print("CALL FUNCTION:", target, args, kwargs)
    def call_method(self, target, *args, **kwargs):
        print("CALL METHOD:", target, args, kwargs)
    def call_module(self, target, *args, **kwargs):
        print("CALL MODULE:", target, args, kwargs)
    def output(self, target, *args, **kwargs):
        print("OUTPUT:", target, args, kwargs)
    def placeholder(self, target, *args, **kwargs):
        print("PLACEHOLDER:", target, args, kwargs)

# TODO: Work out the boxing/aot-autograd nonsense vs using this utility.
#@make_simple_dynamo_backend
def backend(gm: torch.fx.GraphModule, example_inputs):
    gm.print_readable()
    for n in gm.graph.nodes:
        print(n.format_node())
        print(n.meta)
    graph = GraphLowering(gm)
    print(graph)
    graph.run(*example_inputs)

    #gm.print_readable()
    #jit_mod = torch.jit.trace(gm, example_inputs)
    #print("JITMOD:", jit_mod)
    #print(example_inputs)
    # imp = ScriptImporter(text_mode=True)
    # input_module = InputModule(imp(gm, example_inputs))
    # print("INPUT MODULE:", input_module)
    # device_state = _get_device_state()
    # exe = JittableExecutable(
    #     input_module,
    #     device_state,
    #     compiler_flags=("--iree-hal-target-backends=llvm-cpu",),
    # )
    #return exe
    return gm.forward  # return a python callable


# IREE runtime globals. For the CPU right now, there is no device selection,
# so it is easy.
# @functools.lru_cache(maxsize=None)
# def _get_device_state() -> DeviceState:
#     return DeviceState(driver="local-task")
