# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

from iree.compiler.ir import Context
from shark_turbine.aot import *
from turbine_models.custom_models.sd_inference import utils
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor
from turbine_models.turbine_tank import turbine_tank

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="CompVis/stable-diffusion-v1-4",
)
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument("--external_weight_path", type=str, default="")
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir/vmfb without global weights for size and readability, options [safetensors]",
)
parser.add_argument("--device", type=str, default="cpu", help="cpu, cuda, vulkan, rocm")
# TODO: Bring in detection for target triple
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")

class PromptEncoderModule(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        precision,
        hf_auth_token=None,
        do_classifier_free_guidance=True,
    ):
        super().__init__()
        self.torch_dtype = torch.float16 if precision == "fp16" else torch.float32
        self.text_encoder_model_1 = CLIPTextModel.from_pretrained(
            hf_model_name,
            subfolder="text_encoder",
            token=hf_auth_token,
        )
        self.text_encoder_model_2 = CLIPTextModelWithProjection.from_pretrained(
            hf_model_name,
            subfolder="text_encoder_2",
            token=hf_auth_token,
        )
        self.do_classifier_free_guidance = do_classifier_free_guidance

    def forward(
        self, text_input_ids_1, text_input_ids_2, uncond_input_ids_1, uncond_input_ids_2
    ):
        with torch.no_grad():
            prompt_embeds_1 = self.text_encoder_model_1(
                text_input_ids_1,
                output_hidden_states=True,
            )
            prompt_embeds_2 = self.text_encoder_model_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            neg_prompt_embeds_1 = self.text_encoder_model_1(
                uncond_input_ids_1,
                output_hidden_states=True,
            )
            neg_prompt_embeds_2 = self.text_encoder_model_2(
                uncond_input_ids_2,
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds_2[0]
            neg_pooled_prompt_embeds = neg_prompt_embeds_2[0]

            prompt_embeds_list = [
                prompt_embeds_1.hidden_states[-2],
                prompt_embeds_2.hidden_states[-2],
            ]
            neg_prompt_embeds_list = [
                neg_prompt_embeds_1.hidden_states[-2],
                neg_prompt_embeds_2.hidden_states[-2],
            ]

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            neg_prompt_embeds = torch.concat(neg_prompt_embeds_list, dim=-1)

            bs_embed, seq_len, _ = prompt_embeds.shape

            prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(
                bs_embed * 1, -1
            )
            add_text_embeds = pooled_prompt_embeds
            if self.do_classifier_free_guidance:
                neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(1, 1).view(
                    1, -1
                )
                neg_prompt_embeds = neg_prompt_embeds.repeat(1, 1, 1)
                neg_prompt_embeds = neg_prompt_embeds.view(bs_embed * 1, seq_len, -1)
                prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat(
                    [neg_pooled_prompt_embeds, add_text_embeds], dim=0
                )

            add_text_embeds = add_text_embeds.to(self.torch_dtype)
            prompt_embeds = prompt_embeds.to(self.torch_dtype)
            return prompt_embeds, add_text_embeds

def export_clip_model(
    hf_model_name,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target_triple=None,
    max_alloc=None,
    upload_ir=False,
):
    input_len = 77
    if "google/t5" in hf_model_name:
        from transformers import T5Tokenizer, T5Model

        tokenizer = T5Tokenizer.from_pretrained(hf_model_name)
        text_encoder_model = T5Model.from_pretrained(hf_model_name)
        input_len = 512

    else:
        # TODO: Add better filtering mechanism for things that require CLIPProcessor
        if "openai" in hf_model_name:
            tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            hf_subfolder = ""  # CLIPProcessor does not have a subfolder
            input_len = 10
        else:
            # Load the tokenizer and text encoder to tokenize and encode the text.
            tokenizer = CLIPTokenizer.from_pretrained(
                hf_model_name,
                subfolder="tokenizer",
                token=hf_auth_token,
            )
            hf_subfolder = "text_encoder"

        text_encoder_model = PromptEncoderModule(
            hf_model_name,
            subfolder=hf_subfolder,
            token=hf_auth_token,
        )

    mapper = {}
    utils.save_external_weights(
        mapper, text_encoder_model, external_weights, external_weight_path
    )

    if "google/t5" in hf_model_name:

        class CompiledClip(CompiledModule):
            if external_weights:
                params = export_parameters(
                    text_encoder_model,
                    external=True,
                    external_scope="",
                    name_mapper=mapper.get,
                )
            else:
                params = export_parameters(text_encoder_model)

            def main(
                self,
                inp=AbstractTensor(1, input_len, dtype=torch.int64),
                decoder_input_ids=AbstractTensor(1, input_len, dtype=torch.int64),
            ):
                return jittable(text_encoder_model.forward)(
                    input_ids=inp, decoder_input_ids=decoder_input_ids
                )

    else:

        class CompiledClip(CompiledModule):
            if external_weights:
                params = export_parameters(
                    text_encoder_model,
                    external=True,
                    external_scope="",
                    name_mapper=mapper.get,
                )
            else:
                params = export_parameters(text_encoder_model)

            def main(self, inp=AbstractTensor(1, input_len, dtype=torch.int64)):
                return jittable(text_encoder_model.forward)(input_ids=inp)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledClip(context=Context(), import_to=import_to)

    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = utils.create_safe_name(hf_model_name, "-clip")
    if upload_ir:
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(module_str)
        model_name_upload = hf_model_name.replace("/", "_")
        model_name_upload += "-clip"
        blob_name = turbine_tank.uploadToBlobStorage(
            str(os.path.abspath(f"{safe_name}.mlir")),
            f"{model_name_upload}/{model_name_upload}.mlir",
        )
    if compile_to != "vmfb":
        return module_str, tokenizer
    else:
        utils.compile_to_vmfb(module_str, device, target_triple, max_alloc, safe_name)
        if upload_ir:
            return blob_name


if __name__ == "__main__":
    args = parser.parse_args()
    mod_str, _ = export_clip_model(
        args.hf_model_name,
        args.hf_auth_token,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.vulkan_max_allocation,
    )
    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
