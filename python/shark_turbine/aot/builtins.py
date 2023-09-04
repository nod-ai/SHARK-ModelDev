# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tracing builtins."""

import torch._dynamo as dynamo

__all__ = [
    "jittable",
]


class jittable:
    """Decorator which takes a PyTorch function and makes it callable from tracing.

    It will be internally JIT-ed and exported into the module as needed.
    """

    def __init__(self, wrapped_f, *, decomposition_table=None, constraints=None):
        self.wrapped_f = wrapped_f
        self.exported_f = dynamo.export(
            wrapped_f,
            aten_graph=True,
            decomposition_table=decomposition_table,
            constraints=constraints,
            assume_static_by_default=True,
            # TODO: Need to do the signature/tree recomposition ourselves.
            same_signature=False,
        )

    def __repr__(self):
        return f"<Jittable PyTorch func: {self.exported_f}>"
