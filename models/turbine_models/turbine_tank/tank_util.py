# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import iree.compiler as ireec
import torch
from turbine_models.turbine_tank import tank_util
from turbine_models.model_runner import vmfbRunner
from turbine_models.custom_models.sd_inference import utils
from iree import runtime as ireert
import os
from shark_turbine.aot import *
from iree.compiler.ir import Context
from turbine_models.turbine_tank import turbine_tank

torch.manual_seed(0)

BATCH_SIZE = 1

model_list = [
    ("microsoft/resnet-50", "hf_img_cls"),
    ("bert-large-uncased", "hf"),
    ("facebook/deit-small-distilled-patch16-224", "hf_img_cls"),
    ("google/vit-base-patch16-224", "hf_img_cls"),
    ("microsoft/beit-base-patch16-224-pt22k-ft22k", "hf_img_cls"),
    ("microsoft/MiniLM-L12-H384-uncased", "hf"),
    ("google/mobilebert-uncased", "hf"),
    ("mobilenet_v3_small", "vision"),
    ("nvidia/mit-b0", "hf_img_cls"),
    ("resnet101", "vision"),
    ("resnet18", "vision"),
    ("resnet50", "vision"),
    ("squeezenet1_0", "vision"),
    ("wide_resnet50_2", "vision"),
    ("mnasnet1_0", "vision"),
    ("t5-base", "hf_seq2seq"),  # iree-compile failure
    ("t5-large", "hf_seq2seq"),  # iree-compile failure
    ("openai/whisper-base", "hf_causallm"),
    ("openai/whisper-small", "hf_causallm"),
    ("openai/whisper-medium", "hf_causallm"),
    ("facebook/opt-350m", "hf"),
    ("facebook/opt-1.3b", "hf"),
    ("BAAI/bge-base-en-v1.5", "hf"),
    ("facebook/bart-large", "hf_seq2seq"),  # iree-compile fails
    ("gpt2", "hf"),  # iree-compile fails
    ("gpt2-xl", "hf"),  # iree-compile fails
    ("lmsys/vicuna-13b-v1.3", "hf"),
    ("microsoft/phi-1_5", "hf_causallm"),  # nan error reported (correctness issue)
    ("microsoft/phi-2", "hf_causallm"),  # nan error reported (correctness issue)
    ("mosaicml/mpt-30b", "hf_causallm"),  # iree-compile fails
    ("stabilityai/stablelm-3b-4e1t", "hf_causallm"),
]


##################### Hugging Face Image Classification Models ###################################
from transformers import AutoModelForImageClassification
from transformers import AutoFeatureExtractor
from PIL import Image
import requests


def preprocess_input_image(model_name):
    # from datasets import load_dataset
    # dataset = load_dataset("huggingface/cats-image")
    # image1 = dataset["test"]["image"][0]
    # # print("image1: ", image1) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7FA0B86BB6D0>
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=640x480 at 0x7FA0B86BB6D0>
    image = Image.open(requests.get(url, stream=True).raw)
    # feature_extractor = img_models_fe_dict[model_name].from_pretrained(
    #     model_name
    # )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    inputs = feature_extractor(images=image, return_tensors="pt")
    # inputs = {'pixel_values': tensor([[[[ 0.1137..., -0.2000, -0.4275, -0.5294]]]])}
    #           torch.Size([1, 3, 224, 224]), torch.FloatTensor

    return inputs[str(*inputs)]


class HuggingFaceImageClassification(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            output_attentions=False,  # Whether the model returns attentions weights.
            return_dict=False,  # https://github.com/huggingface/transformers/issues/9095
            torchscript=True,
        )

    def forward(self, inputs):
        return self.model.forward(inputs)[0]


def get_hf_img_cls_model(name, import_args):
    model = HuggingFaceImageClassification(name)
    # you can use preprocess_input_image to get the test_input or just random value.
    test_input = preprocess_input_image(name)
    # test_input = torch.FloatTensor(1, 3, 224, 224).uniform_(-1, 1)
    # print("test_input.shape: ", test_input.shape)
    # test_input.shape:  torch.Size([1, 3, 224, 224])
    test_input = test_input.repeat(int(import_args["batch_size"]), 1, 1, 1)
    actual_out = model(test_input)
    # actual_out.shape：  torch.Size([1, 1000])
    return model, test_input, actual_out


##################### Hugging Face LM Models ###################################


class HuggingFaceLanguage(torch.nn.Module):
    def __init__(self, hf_model_name):
        super().__init__()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import transformers as trf

        transformers_path = trf.__path__[0]
        hf_model_path = f"{transformers_path}/models/{hf_model_name}"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_name,  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )
        self.model.config.pad_token_id = None

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


def get_hf_model(name, import_args):
    model = HuggingFaceLanguage(name)
    test_input = torch.randint(2, (int(import_args["batch_size"]), 128))
    actual_out = model(test_input)
    return model, test_input, actual_out


##################### Hugging Face Seq2SeqLM Models ###################################

# We use a maximum sequence length of 512 since this is the default used in the T5 config.
T5_MAX_SEQUENCE_LENGTH = 512


class HFSeq2SeqLanguageModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoTokenizer, T5Model

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenization_kwargs = {
            "pad_to_multiple_of": T5_MAX_SEQUENCE_LENGTH,
            "padding": True,
            "return_tensors": "pt",
        }
        self.model = T5Model.from_pretrained(model_name, return_dict=True)

    def preprocess_input(self, text):
        return self.tokenizer(text, **self.tokenization_kwargs)

    def forward(self, input_ids, decoder_input_ids):
        return self.model.forward(input_ids, decoder_input_ids=decoder_input_ids)[0]


def get_hf_seq2seq_model(name, import_args):
    m = HFSeq2SeqLanguageModel(name)
    encoded_input_ids = m.preprocess_input(
        "Studies have been shown that owning a dog is good for you"
    ).input_ids
    decoder_input_ids = m.preprocess_input("Studies show that").input_ids
    decoder_input_ids = m.model._shift_right(decoder_input_ids)

    test_input = (encoded_input_ids, decoder_input_ids)
    actual_out = m.forward(*test_input)
    return m, test_input, actual_out


##################### Hugging Face CausalLM Models ###################################
from transformers import AutoTokenizer, AutoModelForCausalLM


def prepare_sentence_tokens(hf_model: str, sentence: str):
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model, token="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"
    )
    return torch.tensor([tokenizer.encode(sentence)])


class HFCausalLM(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,  # The pretrained model name.
            # The number of output labels--2 for binary classification.
            num_labels=2,
            # Whether the model returns attentions weights.
            output_attentions=False,
            # Whether the model returns all hidden-states.
            output_hidden_states=False,
            torchscript=True,
            trust_remote_code=True,
            token="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",
        )
        self.model.eval()

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


def get_hf_causallm_model(name, import_args):
    m = HFCausalLM(name)
    test_input = prepare_sentence_tokens(name, "this project is very interesting")
    actual_out = m.forward(test_input)
    return m, test_input, actual_out


################################################################################

##################### Torch Vision Models    ###################################


class VisionModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.train(False)

    def forward(self, input):
        return self.model.forward(input)


def get_vision_model(torch_model, import_args):
    import torchvision.models as models

    default_image_size = (224, 224)
    modelname = torch_model
    if modelname == "alexnet":
        torch_model = models.alexnet(weights="DEFAULT")
        input_image_size = default_image_size
    if modelname == "resnet18":
        torch_model = models.resnet18(weights="DEFAULT")
        input_image_size = default_image_size
    if modelname == "resnet50":
        torch_model = models.resnet50(weights="DEFAULT")
        input_image_size = default_image_size
    if modelname == "resnet50_fp16":
        torch_model = models.resnet50(weights="DEFAULT")
        input_image_size = default_image_size
    if modelname == "resnet50_fp16":
        torch_model = models.resnet50(weights="DEFAULT")
        input_image_size = default_image_size
    if modelname == "resnet101":
        torch_model = models.resnet101(weights="DEFAULT")
        input_image_size = default_image_size
    if modelname == "squeezenet1_0":
        torch_model = models.squeezenet1_0(weights="DEFAULT")
        input_image_size = default_image_size
    if modelname == "wide_resnet50_2":
        torch_model = models.wide_resnet50_2(weights="DEFAULT")
        input_image_size = default_image_size
    if modelname == "mobilenet_v3_small":
        torch_model = models.mobilenet_v3_small(weights="DEFAULT")
        input_image_size = default_image_size
    if modelname == "mnasnet1_0":
        torch_model = models.mnasnet1_0(weights="DEFAULT")
        input_image_size = default_image_size

    model = VisionModule(torch_model)
    test_input = torch.randn(int(import_args["batch_size"]), 3, *input_image_size)
    actual_out = model(test_input)
    return model, test_input, actual_out


def compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name):
    flags = [
        "--iree-input-type=torch",
        "--mlir-print-debuginfo",
        "--mlir-print-op-on-diagnostic=false",
        "--iree-llvmcpu-target-cpu-features=host",
        "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
        "--iree-stream-resource-index-bits=64",
        "--iree-vm-target-index-bits=64",
        "--iree-flow-inline-constants-max-byte-length=1",
    ]
    if device == "cpu":
        flags.append("--iree-llvmcpu-enable-ukernels=all")
        device = "llvm-cpu"
    elif device == "vulkan":
        flags.extend(
            [
                "--iree-hal-target-backends=vulkan-spirv",
                "--iree-vulkan-target-triple=" + target_triple,
                "--iree-stream-resource-max-allocation-size=" + max_alloc,
            ]
        )
    elif device == "rocm":
        flags.extend(
            [
                "--iree-hal-target-backends=rocm",
                "--iree-rocm-target-chip=" + target_triple,
                "--iree-rocm-link-bc=true",
                "--iree-rocm-bc-dir=/opt/rocm/amdgcn/bitcode",
                "--iree-vm-bytecode-module-strip-source-map=true",
                "--iree-opt-strip-assertions=true",
                "--iree-vm-target-truncate-unsupported-floats",
            ]
        )
    elif device == "cuda":
        flags.extend(
            [
                "--iree-hal-target-backends=cuda",
                "--iree-hal-cuda-llvm-target-arch=" + target_triple,
                "--iree-vm-bytecode-module-strip-source-map=true",
                "--iree-vm-target-truncate-unsupported-floats",
            ]
        )
    else:
        print("incorrect device: ", device)

    flatbuffer_blob = ireec.compile_str(
        module_str,
        target_backends=[device],
        extra_args=flags,
    )
    with open(f"{safe_name}.vmfb", "wb+") as f:
        f.write(flatbuffer_blob)
    print("Saved to", safe_name + ".vmfb")


def classic_flow(model, model_name, input, out, run_e2e, expected_err):
    vmfb_name = model_name.replace("/", "_") + ".vmfb"
    model.get_compiled_module(save_to=vmfb_name)

    # if model is not supposed to run e2e, exit at this point (mlir has been uploaded)
    if run_e2e is False:
        assert expected_err > 0
        return

    # run inference using iree runtime
    runner = vmfbRunner("local-task", vmfb_name)
    inputs = [ireert.asdevicearray(runner.config.device, input)]
    keys = list(runner.ctx.modules)
    key = keys[len(keys) - 1]
    results = runner.ctx.modules.__getattr__(key)["main"](*inputs)
    err = utils.largest_error(out.cpu().detach().numpy(), results)
    # cleanup
    os.remove(vmfb_name)
    # accuracy
    assert err < expected_err


def param_flow(model, model_name, model_type, input, out, run_e2e, expected_err):
    weight_name = model_name.replace("/", "_") + ".safetensors"
    mapper = {}
    utils.save_external_weights(mapper, model.model, "safetensors", weight_name)

    # seq2seq models differs from rest as it take two inputs (input_ids, decoder_input_ids)
    if model_type == "hf_seq2seq":

        class Seq2SeqModule(CompiledModule):
            params = export_parameters(
                model.model, external=True, external_scope="", name_mapper=mapper.get
            )

            def main(
                self,
                inp1=AbstractTensor(*(input[0].shape), dtype=input[0].dtype),
                inp2=AbstractTensor(*(input[1].shape), dtype=input[1].dtype),
            ):
                return jittable(model.model.forward)(inp1, inp2)

        inst = Seq2SeqModule(context=Context(), import_to="IMPORT")
        module_str = str(CompiledModule.get_mlir_module(inst))
    else:

        class GlobalModule(CompiledModule):
            params = export_parameters(
                model.model, external=True, external_scope="", name_mapper=mapper.get
            )

            def main(self, inp=AbstractTensor(*input.shape, dtype=input.dtype)):
                return jittable(model.model.forward)(inp)

        inst = GlobalModule(context=Context(), import_to="IMPORT")
        module_str = str(CompiledModule.get_mlir_module(inst))

    mlir_name = model_name.replace("/", "_") + ".mlir"
    with open(mlir_name, "w+") as f:
        f.write(module_str)

    model_name_upload = model_name.replace("/", "_")
    turbine_tank.uploadToBlobStorage(
        str(os.path.abspath(mlir_name)),
        f"{model_name_upload}/{model_name_upload}-params.mlir",
    )

    os.remove(mlir_name)

    if run_e2e is False:
        assert expected_err > 0
        return

    vmfb_name = model_name.replace("/", "_")
    tank_util.compile_to_vmfb(module_str, "cpu", "", "", vmfb_name)

    # run inference using iree runtime
    runner = vmfbRunner("local-task", vmfb_name + ".vmfb", weight_name)
    inputs = [ireert.asdevicearray(runner.config.device, input)]
    keys = list(runner.ctx.modules)
    key = keys[len(keys) - 1]
    results = runner.ctx.modules.__getattr__(key)["main"](*inputs)
    err = utils.largest_error(out.cpu().detach().numpy(), results)

    # clean up
    os.remove(vmfb_name + ".vmfb")
    os.remove(weight_name)

    # accuracy
    assert err < expected_err
