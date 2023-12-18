from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from shark_turbine.aot import *
import iree.runtime as rt

# Loading feature extractor and pretrained model from huggingface
extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

# define a function to do inference
# this will get passed to the compiled module as a jittable function
def forward(pixel_values_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model.forward(pixel_values_tensor).logits
    predicted_id = torch.argmax(logits, -1)
    return predicted_id

# a dynamic module for doing inference
# this will be compiled AOT to a memory buffer
class RN18(CompiledModule):
    params = export_parameters(model)

    def forward(self, x=AbstractTensor(None, 3, 224, 224, dtype=torch.float32)):
        # set a constraint for the dynamic number of batches
        # interestingly enough, it doesn't seem to limit BATCH_SIZE
        const = [x.dynamic_dim(0) < 16]
        return jittable(forward)(x, constraints=const)

# build an mlir module with 1-shot exporter
exported = export(RN18)
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
def print_labels(id):
    for l in id:
        print(model.config.id2label[l])

# finds discrepancies between id0 and id1
def compare_labels(id0, id1):
    return (id0 != id1).nonzero(as_tuple=True)

# load some examples and check for discrepancies between
# compiled module and standard inference (forward function)

from datasets import load_dataset
import random

# BATCH_SIZE <= 23442
BATCH_SIZE = 100
dataset = load_dataset("cats_vs_dogs")
# extract a window of examples.
start_index = random.randint(0,23443 - BATCH_SIZE)
end_index = start_index + BATCH_SIZE
image = dataset['train'][start_index:end_index]["image"]
# convert images to a torch.Tensor(BATCH_SIZE,3,224,224)
inputs = extractor(image, return_tensors="pt")
pixel_tensor = inputs.pixel_values

# perform inferene and compare results
print(f"\nRunning inference on {BATCH_SIZE} pet pictures...")
y1 = shark_infer(pixel_tensor)
y2 = forward(pixel_tensor)
print("Inference completed. Here are a few sample labels. \n")
print_labels(y1.to_host()[0:5])
diff = compare_labels(y1,y2)
print(f'\nDiscrepancies between SharkTurbine and Standard inference results: {diff[0].size()[0]}')
