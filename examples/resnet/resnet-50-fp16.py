from torchvision.models import resnet50, ResNet50_Weights
import torch
import numpy as np
from shark_turbine.aot import *
import iree.runtime as rt

# Loading feature extractor and pretrained model from huggingface
# extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = resnet50(weights="DEFAULT")
float_model = model.eval().float()
model = model.eval().half()


# define a function to do inference
# this will get passed to the compiled module as a jittable function
def vision_forward(pixel_values_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model.forward(pixel_values_tensor)
    predicted_id = torch.argmax(logits, -1)
    return predicted_id


def vision_forward_float(pixel_values_tensor: torch.Tensor):
    with torch.no_grad():
        logits = float_model.forward(pixel_values_tensor)
    predicted_id = torch.argmax(logits, -1)
    return predicted_id


# a dynamic module for doing inference
# this will be compiled AOT to a memory buffer
class Resnet50_f16(CompiledModule):
    params = export_parameters(model)

    def forward(self, x=AbstractTensor(None, 3, 224, 224, dtype=torch.float16)):
        # set a constraint for the dynamic number of batches
        # interestingly enough, it doesn't seem to limit BATCH_SIZE
        const = [x.dynamic_dim(0) < 16]
        return jittable(vision_forward)(x, constraints=const)


# build an mlir module with 1-shot exporter
exported = export(Resnet50_f16)
# compile exported module to a memory buffer
compiled_binary = exported.compile(save_to=None)


# return type is rt.array_interop.DeviceArray
# np.array of outputs can be accessed via to_host() method
def shark_infer(x):
    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    y = vmm.forward(x)
    return y


# prints the text corresponding to output label codes
def print_labels(class_id):
    weights = ResNet50_Weights.DEFAULT
    for l in class_id:
        print(weights.meta["categories"][l])


# finds discrepancies between id0 and id1
def largest_error(array1, array2):
    absolute_diff = np.abs(array1 - array2)
    max_error = np.max(absolute_diff)
    return max_error


# load some examples and check for discrepancies between
# compiled module and standard inference (forward function)

x = torch.randn((10, 3, 224, 224), dtype=torch.float16)
x_float = torch.randn((10, 3, 224, 224), dtype=torch.float32)
y0 = shark_infer(x).to_host()
float_model = float_model.float()
y1 = np.asarray(vision_forward_float(x_float))
print_labels(y0)
print(
    f"Largest error between turbine (fp16) and pytorch (fp32) baseline is {largest_error(y0,y1)}"
)
