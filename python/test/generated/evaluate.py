from stats import ErrorAggregatorDict
import logging

from shark_turbine.dynamo.importer import FxImporter
import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd
from torch.fx import (
    GraphModule,
)

from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from typing import List

def default_decompositions():
    return get_decompositions(
        [
            torch.ops.aten.embedding_dense_backward,
            torch.ops.aten.native_layer_norm_backward,
            torch.ops.aten.slice_backward,
            torch.ops.aten.select_backward,
            torch.ops.aten.norm.ScalarOpt_dim,
            torch.ops.aten.native_group_norm,
            torch.ops.aten.upsample_bilinear2d.vec,
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes,
            torch.ops.aten.native_layer_norm,
            torch.ops.aten.masked_fill.Tensor,
            torch.ops.aten.masked_fill.Scalar,
            torch.ops.aten._native_batch_norm_legit_functional,
            torch.ops.aten.squeeze.dims,
        ]
    )


def create_backend():
    imp = FxImporter()

    def import_compiler(gm: GraphModule, example_inputs):
        gm = make_fx(
            functionalize(gm),
            decomposition_table=default_decompositions(),
        )(*example_inputs)

        try:
            imp.import_graph_module(gm)
        finally:
            pass
        imp.module.operation.verify()
        return gm

    backend = import_compiler
    backend = aot_autograd(fw_compiler=backend)
    return backend


def evaluate_importer(nn_cls, get_init_args, get_forward_args, test_identifier):
    log = logging.getLogger("turbine-test")
    try:
        args, kwargs = get_init_args()
        nn_module = nn_cls(*args, **kwargs)
        opt_mod = torch.compile(nn_module, backend=create_backend())

        fargs, fkwargs = get_forward_args()
        opt_mod(*fargs, **fkwargs)
    except Exception as e:
        err = ErrorAggregatorDict.single(str(e), test_identifier)
        return err
