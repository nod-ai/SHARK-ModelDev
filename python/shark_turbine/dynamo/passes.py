import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from torch.func import functionalize
from typing import Dict, List

# default decompositions pulled from SHARK
DEFAULT_DECOMPOSITIONS = [
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
    torch.ops.aten.t,
    torch.ops.aten.addmm,
]


CPU_DECOMPOSITIONS = [
    # decompositions that aid us in handling nn.BatchNorm2d
    torch.ops.aten._native_batch_norm_legit_functional,
    torch.ops.aten._native_batch_norm_legit.no_stats,
    torch.ops.aten.squeeze.dims,
    # decompositions for miscellaneous ops that are not handled in torch-mlir but have available decompositions
    torch.ops.aten.soft_margin_loss,
    torch.ops.aten.im2col,
    torch.ops.aten._euclidean_dist,
    torch.ops.aten.index_copy,
    torch.ops.aten.index_copy_,
    torch.ops.aten.grid_sampler_2d,
    torch.ops.aten.log_sigmoid_forward,
    torch.ops.aten.unsafe_split.Tensor,
    torch.ops.aten.binary_cross_entropy,
    torch.ops.aten.dot,
    torch.ops.aten._adaptive_avg_pool2d,
    torch.ops.aten._prelu_kernel,
    torch.ops.aten.full,
    torch.ops.aten._log_softmax,
    torch.ops.aten.nll_loss_forward,
    torch.ops.aten._to_copy,
]

def gptq_transform(fx_g):
    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            print(dir(torch.ops.constant))
           # if node.kwargs.get("device") == torch.device(device="cuda:0"):
            if node.target in [torch.ops.prims.device_put.default, torch.ops.prims.device_put]:
                print("before changing graph")            
                fx_g.print_readable()
                prev_node = node.all_input_nodes[0]
                prev_node_kwargs = prev_node.kwargs.copy()
                print("kwargs = ", prev_node_kwargs)
                print(dir(prev_node))
                print("setting prev_node: ", prev_node.name)
                prev_node.next.prepend(node.next)
                print("prev_node next: ", prev_node.next)
                i = 0
                for n in node.next.all_input_nodes:
                    if n == node:
                        print("node next: ", node.next.name)
                        print("setting node next input: ", node.next.all_input_nodes[i].name)
                        node.next.all_input_nodes[i].prepend(prev_node)
                    i += 1
                print("after changing graph")
                fx_g.print_readable()
                fx_g.graph.erase_node(node)
                '''
                print("node: ", dir(node))
                print("node inputs: ", node.all_input_nodes)
                print("node next: ", node.next)
                #print("node target: ", node._pretty_print_target())#target.name)#dir(node.target))
                print("graph: ", dir(fx_g.graph))
                tracer = torch.fx.proxy.GraphAppendingTracer(node.graph)
                fx_g.print_readable()
                with node.graph.inserting_before(node):
                    proxy_args = torch.fx.node.map_arg(node.args, lambda x: torch.fx.Proxy(x, tracer))
                    proxy_kwargs = torch.fx.node.map_arg(node.kwargs, lambda x: torch.fx.Proxy(x, tracer))
                    output_proxy = node.target(*proxy_args, **proxy_kwargs)
                    print("creating node: ", output_proxy.node)
                    node.replace_all_uses_with(output_proxy.node)
                    print("erasing node: ", node.name)
                    fx_g.graph.erase_node(node)
                print("after removeal")
                fx_g.recompile()
                fx_g.print_readable()
                '''
            elif node.kwargs.get("device") == torch.device(device="cuda:0"):
                updated_kwargs = node.kwargs.copy()
                updated_kwargs["device"] = torch.device(device="cpu")
                node.kwargs = updated_kwargs
    fx_g.graph.eliminate_dead_code()
    fx_g.recompile()

def apply_decompositions(
    gm: torch.fx.GraphModule,
    example_inputs,
    decompose_ops: List[torch._ops.OpOverload] = None,
):
    if decompose_ops is None:
        return gm

    decompositions = get_decompositions(decompose_ops)
    print("before make_fx gm:\n", gm)
    gm = make_fx(
        functionalize(gm),
        decomposition_table=decompositions,
    )(*example_inputs)
#    print("after make_fx")
#    gm.half()
#    print("after half")
    gptq_transform(gm)
    print("after make_fx gm:\n", gm)
#    gm.graph.lint()
#    gm.recompile()

    return gm


def turbine_cpu_pass_pipeline(gm: torch.fx.GraphModule, example_inputs):
    decompose_ops = DEFAULT_DECOMPOSITIONS + CPU_DECOMPOSITIONS
    return apply_decompositions(gm, example_inputs, decompose_ops)
