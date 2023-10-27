import os
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from shark_turbine.aot import *
from iree.compiler.ir import Context
from iree.compiler.api import Session
from datasets import load_dataset
import PIL

# Load feature extractor and pretrained model from huggingface
extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")

###
#standard inference example (no SHARK-Turbine):
###

#load an example
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

#image.save("cats-image.jpg") #if you want to see the cute picture :)

#uncomment the line below to run your own image through inference.
#image = PIL.JpegImagePlugin.JpegImageFile("yourexamplepicture.jpg")

#extract features from image to feed to model
inputs = extractor(image, return_tensors="pt")
pixel_tensor = inputs.pixel_values

#standard torch application of model to extracted feature input.
def std_infer(pixel_values_tensor: torch.Tensor):
    with torch.no_grad():
        logits = model.forward(pixel_values_tensor).logits
    predicted_label = torch.argmax(logits,-1).item() 
    textout = model.config.id2label[predicted_label]
    return textout

###
#Using SHARK-Turbine to write a torch-mlir file for performing inference with the model.
###

class wtfamidoing2(CompiledModule):
    #this one is trying to pass the pixel_value tensor as an Abstract Tensor?
    params = export_parameters(model, initialize = True) #         is this necessary? 

    def forward(self, x = AbstractTensor(1, 3, 224, 224, dtype=torch.float32) ):
        with torch.no_grad():
            logits0 = jittable(model.forward)(x)['logits']
        label = self.labelout(logits0)
        #str_output = model.config.id2label[predicted_label0] 
        return label

    @jittable
    def labelout(logitsin):
        label =  torch.argmax(logitsin,-1)
        return label

#build an instance of our CompiledModule and get the corresponding mlir code
inst2 = wtfamidoing2(context=Context(), import_to="IMPORT")

#module_str2 = str(CompiledModule.get_mlir_module(inst2))
#print(module_str2)

session = Session()
ExportedModule = exporter.ExportOutput(session,inst2)
compiled_binary = ExportedModule.compile(save_to=None)

def shark_infer(x):
    import iree.runtime as rt
    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    y = vmm.main(x)
    return y