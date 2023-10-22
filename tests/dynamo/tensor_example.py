# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import time
import unittest

import numpy as np
import torch
from viztracer import VizTracer


logging.basicConfig(level=logging.DEBUG)
# Public API imports.
from shark_turbine.dynamo import TurbineMode, enable, disable

enable()

def unary():
    t1 = -5*torch.ones(2, 3)
    t1 = t1.to(device="turbine")
    t2 = torch.abs(t1)
    print(t2.cpu())
    return t2

def binary():
    t1 = 5*torch.ones(2, 3)
    t1 = t1.to(device="turbine")
    # mm = torch.matmul(t1, t2)
    for _ in range(10):
        t1 = t1 + 4
    print(t1.cpu())

def matmul():
    t1 = (5*torch.ones(2, 3)).to(device="turbine")
    t2 = (3*torch.ones(3, 2)).to(device="turbine")
    t3 = torch.matmul(t1, t2)
    print(t3.cpu())
    return t3

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = torch.nn.Linear(64, 32, bias=True)
        self.layer1 = torch.nn.Linear(32, 16, bias=True)
        self.layer2 = torch.nn.Linear(16, 7, bias=True)
        self.layer3 = torch.nn.Linear(7, 7, bias=True)

    def forward(self, x: torch.Tensor):
        x = self.layer0(x)
        x = torch.sigmoid(x)
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        x = torch.sigmoid(x)
        x = self.layer3(x)
        return x

def MLP_run():
    m = MLP()
    input = torch.randn(16, 64)
    iter = 100
    start = time.time()
    with torch.no_grad():
        for i in range(iter):
            ref_out = m(input)
    end = time.time()
    print(f"Regular speed: {iter/(end-start)} it / sec")
    print(ref_out)
    m.to("turbine")
    input = input.to("turbine")
    turbine_output = m(input)
    start = time.time()
    # tracer = VizTracer()
    with torch.no_grad():
        # tracer.start()
        for i in range(iter):
            turbine_output = m(input)
        # tracer.stop()
    end = time.time()
    # tracer.save("turbine_run.json")
    print(f"Turbine speed: {iter/(end-start)} it / sec")
    print(turbine_output.cpu())
    return

def linear():
    m = torch.nn.Linear(20, 30)
    input = torch.randn(128, 20)
    m.to("turbine")
    d_input = input.to("turbine")
    iter = 10
    start = time.time()
    for i in range(iter):
        output = m(d_input)
    end = time.time()
    print(f"{10/(end-start)} it / sec")
    print(output.cpu())

if __name__ == "__main__":
    MLP_run()
