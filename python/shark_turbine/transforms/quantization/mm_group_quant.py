# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .. import rewriter

from iree.compiler.ir import (
    Operation,
)


class TransposedMMResult(rewriter.OpMatchResult):
    def __init__(self, op: Operation, weight_global: Operation, param_name: str):
        super().__init__(op)
        self.weight_global = weight_global
        self.param_name = param_name

    def __repr__(self):
        return f"TransposedMM(weight={self.param_name}, {self.op})"


class TransposedMMMatcher(rewriter.NamedOpMatcher):
    def __init__(self, globals: rewriter.GlobalsDict):
        super().__init__("torch.aten.mm")
        self.globals = globals

    def match(self, op: Operation):
        weight_transpose = rewriter.Transpose2DMatcher()(op.operands[1])
        if not weight_transpose:
            return None
        weight_load = rewriter.GlobalLoadMatcher(self.globals)(weight_transpose.input)
        if not weight_load or not weight_load.resolved_global:
            return None

        return TransposedMMResult(
            op, weight_load.resolved_global, weight_load.global_ref
        )


class MMGroupQuantRewriter(rewriter.Pass):
    def run(self):
        globals = self.globals
        mms = rewriter.match_children(self.funcs, TransposedMMMatcher(globals))
        print(mms)


if __name__ == "__main__":
    rewriter.main(MMGroupQuantRewriter)
