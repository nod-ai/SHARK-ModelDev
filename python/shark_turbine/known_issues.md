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

There is an existing issue in PyTorch that is tracking this problem in the `aot-eager` backend: https://github.com/pytorch/pytorch/issues/105327.
This issue arises because this particular op is not handled in the PyTorch dispatch logic, and is instead suppresed [here](https://github.com/pytorch/pytorch/blob/ddf36c82b83b2db3be7ce7a85d4aea3507c9d7ef/torch/_dispatch/python.py#L108)

Note that during the test, there wasn't nop_decomposition called.

### Autograd failure in `aten.lift` in the aot_eager, inductor, and turbine backend.
`aten.lift` is decomposed to `aten.alias`, and then to `view_of`.
However, AOTAutograd seems to fail in collecting metadata on function when using `aot-eager`, `dynamo`, and `turbine` backend.
```
RuntimeError: !at::functionalization::impl::isFunctionalTensor(self) 
INTERNAL ASSERT FAILED at "../aten/src/ATen/FunctionalizeFallbackKernel.cpp":167, please report a bug to PyTorch. 
```

### No constants for fake input assertion error in `aten.lift_fresh` and `aten.lift_fresh_copy`
`lift_fresh` is ONLY called by `torch.Tensor()` call according to native_functions.yaml.
During dispatch, calling `torch.tensor(x)` will return an empty faketensors (`flat_arg_fake_tensors = []`),
while while directly calling `lift_fresh` or `lift_fresh_copy` returns faketensor inside faketensors.
(e.g. `[FakeTensor(FakeTensor(..., device='meta', size=(1,)), cpu)]`).

During the test, there wasn't `nop_decomposition(...)` called contary to test case of calling `aten.lift` directly.
nop_decomposition() decomposes `aten.detach`, `aten.lift`, `aten.lift_fresh` into `aten.alias`.

This is because `aten.lift_fresh` and `aten.lift_fresh_copy` are functionalized here.
There's no decomposition of functionalized `aten.lift_fresh` and `aten.lift_fresh_copy` currently.
When using `inductor`, these ops are transformed to `aten.clone` and lowered down to IR.

TODO: Figure out how `aten.lift_fresh` and `aten.lift_fresh_copy` works in `eager` mode.

```python
def basic_3(x):
    return torch.tensor(x)
    # return torch.ops.aten.lift_fresh(x) # this returns some value inside the faketensor
```

There is also difference between the latest torch and nightly.
In torch, directly calling `aten.lift_fresh` leads to recursion error.
```
`lift_fresh: RecursionError: maximum recursion depth exceeded while calling a Python object`
```
In the nightly build, recursion error is fixed by passing in `const args/kwargs`,
but throws an assertion ero saying that fake inputs should have constants.
```
AssertionError: f{func} should not have fake inputs without constants
```

