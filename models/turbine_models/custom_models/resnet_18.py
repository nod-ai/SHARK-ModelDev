import os
import sys
import re

from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from shark_turbine.aot import *
from iree.compiler.ir import Context
import iree.runtime as rt
from turbine_models.custom_models.sd_inference import utils

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="microsoft/resnet-18",
)
parser.add_argument("--run_vmfb", action="store_true")
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument("--vmfb_path", type=str, default="")
parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, vulkan, rocm")
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")

# TODO: Add other resnet models


class Resnet18Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-18"
        )
        # self.extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")

    def forward(self, pixel_values_tensor: torch.Tensor):
        with torch.no_grad():
            logits = self.model.forward(pixel_values_tensor).logits
        predicted_id = torch.argmax(logits, -1)
        return predicted_id


def export_resnet_18_model(
    resnet_model, compile_to="torch", device=None, target_triple=None, max_alloc=None
):
    class CompiledResnet18Model(CompiledModule):
        params = export_parameters(resnet_model.model)

        def main(self, x=AbstractTensor(None, 3, 224, 224, dtype=torch.float32)):
            const = [x.dynamic_dim(0) < 16]
            return jittable(resnet_model.forward)(x, constraints=const)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledResnet18Model(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    if compile_to != "vmfb":
        return module_str
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, "resnet_18")


def run_resnet_18_vmfb_comparison(resnet_model, args):
    config = rt.Config(args.device)

    if args.vmfb_path:
        mod = rt.VmModule.mmap(config.vm_instance, args.vmfb_path)
    elif os.path.exists("resnet_18.vmfb"):
        mod = rt.VmModule.mmap(config.vm_instance, "resnet_18.vmfb")
    else:
        sys.exit("no vmfb_path provided, required for run_vmfb")

    vm_modules = [
        mod,
        rt.create_hal_module(config.vm_instance, config.device),
    ]
    ctx = rt.SystemContext(
        vm_modules=vm_modules,
        config=config,
    )
    inp = torch.rand(5, 3, 224, 224, dtype=torch.float32)
    device_inputs = [rt.asdevicearray(config.device, inp)]

    # Turbine output
    CompModule = ctx.modules.compiled_resnet18_model
    turbine_output = CompModule["main"](*device_inputs)
    print(
        "TURBINE OUTPUT:",
        turbine_output.to_host(),
        turbine_output.to_host().shape,
        turbine_output.to_host().dtype,
    )

    # Torch output
    torch_output = resnet_model.forward(inp)
    torch_output = torch_output.detach().cpu().numpy()
    print("TORCH OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)

    err = utils.largest_error(torch_output, turbine_output)
    print("LARGEST ERROR:", err)
    assert err < 9e-5


if __name__ == "__main__":
    args = parser.parse_args()
    resnet_model = Resnet18Model()
    if args.run_vmfb:
        run_resnet_18_vmfb_comparison(resnet_model, args)
    else:
        mod_str = export_resnet_18_model(
            resnet_model,
            args.compile_to,
            args.device,
            args.iree_target_triple,
            args.vulkan_max_allocation,
        )
        safe_name = "resnet_18"
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(mod_str)
        print("Saved to", safe_name + ".mlir")
