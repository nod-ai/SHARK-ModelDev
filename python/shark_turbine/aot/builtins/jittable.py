# Copyright 2023 Nod Labs, Inc
# Portions Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tracing builtins."""

import sys

import torch._dynamo as dynamo

from ..procedural import (
    CallableIntrinsic,
    ProcedureTrace,
)

from ...dynamo.importer import FxImporter
from ...dynamo.passes import (
    turbine_cpu_pass_pipeline,
)

from iree.compiler.passmanager import (
    PassManager,
)


class jittable(CallableIntrinsic):
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

    def resolve_call(self, proc_trace: ProcedureTrace, *args):
        # Ask dynamo to give us an aten graph.
        gm, guards = self.exported_f(*args)
        # TODO: What to do with kwargs?
        gm = turbine_cpu_pass_pipeline(gm, args)

        # Import the FX graph to MLIR.
        fx_importer = FxImporter(context=proc_trace.context)
        fx_importer.import_graph_module(gm)
        print(fx_importer.module, file=sys.stderr)

        with proc_trace.context:
            pm = PassManager.parse(
                "builtin.module(torch-backend-to-linalg-on-tensors-backend-pipeline,symbol-dce)"
            )
            pm.run(fx_importer.module.operation)
        print(fx_importer.module.operation, file=sys.stderr)

        # TODO: Decide what to do from here:
        #   a. Keep importing into a standalone module and merge vs importing
        #      into the main module.
        #   b. Run torch-mlir conversions incrementally or all at once at the end.
        #   c. Rework the FxImporter to not be based on Module (make it Operation).
        #   e. Introduce proxy tensor objects and convert appropriately.
