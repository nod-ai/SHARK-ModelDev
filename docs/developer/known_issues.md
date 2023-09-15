# Known Issues in SHARK-Turbine

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

## Ephemeral Tensor objects from `aten.lift_fresh_copy.default`
```py
import torch
def forward(self):
    return torch.tensor([1,2])
```
```
# in importer -> import_argument
torch._dynamo.exc.BackendCompilerFailed: compiler_fn raised KeyError: (_tensor_constant0, 0)
torch._dynamo.exc.BackendCompilerFailed: compiler_fn raised AssertionError: Can not create literal tensor for unsupported datatype: torch.complex64
```
This error arises due to an odd case in the Fx Graph generation where the
graph module for our code generates a node `_tensor_constant0 = self._tensor_constant0` with no traceable origin within
the graph. Torch dynamo dynamically creates this attribute in the top level module object, hence this object is never 
passed through our importer, meaning that our lookup for the appropriate MlirValue in the importer's `_v` table fails. This consistently
occurs when the graph generates an intermediate `aten.lift_fresh_copy` as in the case of creating a new tensor above.

We now have a fix for this by directly instantiating the object using a reference to the top level graph module in the 
importer, but this method does not support all torch datatypes - in particular it fails to support `bfloat16` and 
complex datatypes.


## Assertion failure in `aten.lift` in the aot_eager, inductor, and turbine backend.
```python
import torch
def forward(self, x):
    return torch.ops.aten.lift(x)
```
```
RuntimeError: !at::functionalization::impl::isFunctionalTensor(self) 
INTERNAL ASSERT FAILED at "../aten/src/ATen/FunctionalizeFallbackKernel.cpp":167, please report a bug to PyTorch. 
```
[`aten.lift`](https://github.com/pytorch/pytorch/blob/3a3cf0e09d475df9237c95ebd14debf650e0c038/aten/src/ATen/native/native_functions.yaml#L7583) seems to fail the [functionalization stage](https://github.com/pytorch/pytorch/blob/3a3cf0e09d475df9237c95ebd14debf650e0c038/aten/src/ATen/FunctionalizeFallbackKernel.cpp#L176), 
in particular it seems that the input tensor fails an [assertion](https://github.com/pytorch/pytorch/blob/3a3cf0e09d475df9237c95ebd14debf650e0c038/aten/src/ATen/FunctionalTensorWrapper.cpp#L575) that it is of functional form.

[PyTorch Issue](https://github.com/pytorch/pytorch/issues/107961)