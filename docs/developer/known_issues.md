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

## TorchDynamo failure in training backward due to `aten.scalar_tensor` output not wrapped as a fake tensor

```python
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out
```
During the training in backwards,
`aten.where.self` expects fake inputs, but `aten.scalar_tensor` output is not wrapped as a fake tensor.
```
File "/home/brucekimrok/CLionProjects/SHARK-Turbine/tvenv3.11/lib/python3.11/site-packages/torch/_subclasses/fake_tensor.py", line 1632, in validate
raise Exception(
Exception: Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode with 'allow_non_fake_inputs'. Found in aten.where.self(FakeTensor(..., size=(64, 1), dtype=torch.bool), FakeTensor(..., size=(64, 1), dtype=torch.int64), tensor(0, size=()))
```
https://github.com/pytorch/pytorch/blob/98c8550158a4a79c4d39533a5331c5953f6ea279/torch/_subclasses/fake_tensor.py#L1657-L1669

Relevant issue is raised here: [PyTorch Issue](https://github.com/pytorch/pytorch/issues/92941).
However, this case is about when DDP optimization + Dynamo + `aten.where` are invoked.
The [PR](https://github.com/pytorch/pytorch/pull/92986) to address this issue was made in `torch/_dynamo/optimizations/distributed.py`.
In our case, we do not use DDP optimization. 

## FX emitted as None due to bug in TorchScript in `aten.convolution_backward` 

When schema calls for a Tensor, sometimes None is emitted due to the way TS is maintained.  
For `convolution_backward` op, TS has a problem of returning None when output_mask=[True, True, False].  
In eager mode, similar can happen.
https://github.com/pytorch/pytorch/issues/97524

Vivek [fixed movdedim](https://github.com/llvm/torch-mlir/pull/1773) to allow torch-mlir emitted when output_mask=[True, True, True]  
So we should find a way to set Output_mask = [True, True, True] to fix this issue.

