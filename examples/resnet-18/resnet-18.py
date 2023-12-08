import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from shark_turbine.aot import *
from iree.compiler.ir import Context
from iree.compiler.api import Session
import iree.runtime as rt
from datasets import load_dataset

# Loading feature extractor and pretrained model from huggingface
extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

# load an example
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

# if you want to see the cat picture:
# image.save("cats-image.jpg")

# if you want to run a custom image through inference.
# import PIL
# image = PIL.JpegImagePlugin.JpegImageFile("yourexamplepicture.jpg")

# extract features from image to feed to model
inputs = extractor(image, return_tensors="pt")
pixel_tensor = inputs.pixel_values


# define a function to do inference
# this will get passed to the compiled module as a jittable function
def forward(pixel_values_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model.forward(pixel_values_tensor).logits
    predicted_id = torch.argmax(logits, -1)
    return predicted_id


# A dynamic module for doing inference
class RN18(CompiledModule):
    params = export_parameters(model)

    def forward(self, x=AbstractTensor(None, 3, 224, 224, dtype=torch.float32)):
        # set a constraint for the dynamic number of batches
        const = [x.dynamic_dim(0) < 16]
        return jittable(forward)(x, constraints=const)


# build an mlir module to compile with 1-shot exporter
exported = export(RN18)

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


# prints the text labels for output ids
def print_labels(id):
    for num in id:
        print(model.config.id2label[num])


# not sure what the point was of the dynamic dim constraint
# also amusing that random tensors are always jellyfish
x = torch.randn(17, 3, 224, 224)
x[2] = pixel_tensor
y = shark_infer(x)
print_labels(y.to_host())
