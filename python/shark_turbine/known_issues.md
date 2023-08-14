# Known Issues in SHARK-Turbine

## Handling lists of optional types
```py
from torch import nn
class foomod(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x):
        return self.up(x)
```
```
# occuring in importer -> import_list_arguments
compiler_fn raised TypeError: Heterogeneous lists are not supported: expected <class 'NoneType'>, got <class 'torch.fx.node.Node'>
```
An example is attempting to import `nn.Upsample`. This module internally makes a call to `F.interpolate` which eventually 
calls `aten.index.Tensor` whose [second argument](https://github.com/llvm/torch-mlir/blob/50f5b658b6dc50f664d78c89c403149b064fb59b/include/torch-mlir/Dialect/Torch/IR/GeneratedTorchOps.td#L7389C46-L7389C46) is an
optional list of tensors. If indices in a few dimensions are omitted in favor of `None`, we get an error. In reality these values
should have an `AnyTorchOptionalTensorType` type, we need a way to set optional types when importing lists in this scenario.


## Dealing with functional variants of Torch Ops

```py
import torch.nn.functional as F
def forward(self, x):
    return F.max_pool2d(8, x)
```
```
# occuring in importer -> import_list_arguments
compiler_fn raised IndexError: list index out of range
```

Currently, we have issues dealing with functional variants of
torch operations that do not define meaningful defaults for their arguments.
Two common operations for which this issue arises are `F.avg_pool2d` and `F.max_pool2d`.
Taking `max_pool2d` as an example, the [functional version](https://pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html) sets `stride=None` by default (which returns an empty list to the importer), 
however, the actual intended default setting is to set `stride=kernel_size`. This issue does not occur with the corresponding `nn.Module` wrapper `MaxPool2d` because
it actually [manually sets the intended default value](https://pytorch.org/docs/stable/_modules/torch/nn/modules/pooling.html#_MaxPoolNd). The same issue is at play in `avg_pool2d`.


## Ephemeral Tensor objects from `aten.lift_fresh_copy`
```py
def forward(self, x, y):
        x[y == 1] = 2
```
```
# in importer -> import_argument
torch._dynamo.exc.BackendCompilerFailed: compiler_fn raised KeyError: (_tensor_constant0, 0)
```
This error arises due to an odd case in the Fx Graph generation where the
graph module for our code generates a node `_tensor_constant0 = self._tensor_constant0` with no traceable origin within 
the graph. This means that our lookup for the appropriate MlirValue in the importer's `_v` table fails. This consistently 
occurs when the graph generates an intermediate `aten.lift_fresh_copy` as in the boolean indexing example above.

There is a known issue in PyTorch https://github.com/pytorch/pytorch/issues/105327.
```
BackendCompilerFailed: backend='aot_eager' raised:
RuntimeError: !at::functionalization::impl::isFunctionalTensor(self) INTERNAL ASSERT FAILED at "/raid/rzou/pt/debug-cpu3/aten/src/ATen/FunctionalizeFallbackKe
rnel.cpp":191, please report a bug to PyTorch.
```