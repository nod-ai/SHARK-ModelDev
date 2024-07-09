# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import numpy as np
from shark_turbine.aot import *
from shark_turbine.transforms.general.add_metadata import AddMetadataPass

from turbine_models.custom_models.sd_inference import utils
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer


class PromptEncoderModule(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        precision,
        hf_auth_token=None,
        do_classifier_free_guidance=True,
        batch_size=1,
        batch_input=False,
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
        self.do_classifier_free_guidance = True
        self.batch_size = batch_size
        self.batch_input = batch_input

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
            if not self.batch_input:
                prompt_embeds = prompt_embeds.repeat(self.batch_size, 1, 1)
            add_text_embeds = pooled_prompt_embeds
            if not self.batch_input:
                add_text_embeds = add_text_embeds.repeat(self.batch_size, 1)
            if self.do_classifier_free_guidance:
                if not self.batch_input:
                    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(
                        1, 1
                    ).view(1, -1)
                neg_prompt_embeds = neg_prompt_embeds.repeat(1, 1, 1)
                neg_prompt_embeds = neg_prompt_embeds.view(bs_embed * 1, seq_len, -1)
                if not self.batch_input:
                    neg_prompt_embeds = neg_prompt_embeds.repeat(self.batch_size, 1, 1)
                prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
                if not self.batch_input:
                    neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(
                        self.batch_size, 1
                    )
                add_text_embeds = torch.cat(
                    [neg_pooled_prompt_embeds, add_text_embeds], dim=0
                )
            add_text_embeds = add_text_embeds.to(self.torch_dtype)
            prompt_embeds = prompt_embeds.to(self.torch_dtype)
            return prompt_embeds, add_text_embeds

    def forward_turbo(self, text_input_ids_1, text_input_ids_2):
        with torch.no_grad():
            prompt_embeds_1 = self.text_encoder_model_1(
                text_input_ids_1,
                output_hidden_states=True,
            )
            prompt_embeds_2 = self.text_encoder_model_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds_2[0]

            prompt_embeds_list = [
                prompt_embeds_1.hidden_states[-2],
                prompt_embeds_2.hidden_states[-2],
            ]
            # neg_prompt_embeds_list = [
            #     torch.zeros_like(prompt_embeds_list[0]), # dummy tensor
            #     torch.zeros_like(prompt_embeds_list[1]), # dummy tensor
            # ]

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

            bs_embed, seq_len, _ = prompt_embeds.shape

            prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(
                bs_embed * 1, -1
            )
            prompt_embeds = prompt_embeds.repeat(self.batch_size, 1, 1)
            add_text_embeds = pooled_prompt_embeds
            add_text_embeds = add_text_embeds.repeat(self.batch_size, 1)

            add_text_embeds = add_text_embeds.to(self.torch_dtype)
            prompt_embeds = prompt_embeds.to(self.torch_dtype)
            return prompt_embeds, add_text_embeds


def export_prompt_encoder(
    hf_model_name,
    hf_auth_token=None,
    max_length=64,
    batch_size=1,
    precision="fp16",
    compile_to="torch",
    external_weights=None,
    external_weight_path=None,
    device=None,
    target=None,
    ireec_flags=None,
    exit_on_vmfb=True,
    pipeline_dir=None,
    input_mlir=None,
    attn_spec=None,
    weights_only=False,
    batch_input=False,
    decomp_attn=False,  # Compatibility
):
    do_classifier_free_guidance = True

    safe_name = utils.create_safe_name(
        hf_model_name,
        f"_bs{batch_size}_{str(max_length)}-{precision}-prompt-encoder-{device}",
    )
    if pipeline_dir not in [None, ""]:
        safe_name = os.path.join(pipeline_dir, safe_name)

    if input_mlir:
        vmfb_path = utils.compile_to_vmfb(
            input_mlir,
            device,
            target,
            ireec_flags,
            safe_name,
            mlir_source="file",
            return_path=not exit_on_vmfb,
            const_expr_hoisting=True,
            attn_spec=attn_spec,
        )
        return vmfb_path
    # Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        hf_model_name,
        subfolder="tokenizer",
        token=hf_auth_token,
        model_max_length=max_length,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        hf_model_name,
        subfolder="tokenizer_2",
        token=hf_auth_token,
        model_max_length=max_length,
    )
    tokenizers = [tokenizer_1, tokenizer_2]
    prompt_encoder_module = PromptEncoderModule(
        hf_model_name,
        precision,
        hf_auth_token,
        do_classifier_free_guidance,
        batch_size=batch_size,
        batch_input=batch_input,
    )

    input_batchsize = 1
    if batch_input:
        input_batchsize = batchsize

    if precision == "fp16":
        prompt_encoder_module = prompt_encoder_module.half()
    mapper = {}

    utils.save_external_weights(
        mapper, prompt_encoder_module, external_weights, external_weight_path
    )

    if weights_only:
        return None, external_weight_path

    class CompiledClip(CompiledModule):
        if external_weights:
            params = export_parameters(
                prompt_encoder_module,
                external=True,
                external_scope="",
                name_mapper=mapper.get,
            )
        else:
            params = export_parameters(prompt_encoder_module)

        def encode_prompts(
            self,
            t_ids_1=AbstractTensor(input_batchsize, max_length, dtype=torch.int64),
            t_ids_2=AbstractTensor(input_batchsize, max_length, dtype=torch.int64),
            uc_ids_1=AbstractTensor(input_batchsize, max_length, dtype=torch.int64),
            uc_ids_2=AbstractTensor(input_batchsize, max_length, dtype=torch.int64),
        ):
            return jittable(prompt_encoder_module.forward)(
                t_ids_1, t_ids_2, uc_ids_1, uc_ids_2
            )

        def encode_prompts_turbo(
            self,
            t_ids_1=AbstractTensor(input_batchsize, max_length, dtype=torch.int64),
            t_ids_2=AbstractTensor(input_batchsize, max_length, dtype=torch.int64),
        ):
            return jittable(prompt_encoder_module.forward_turbo)(t_ids_1, t_ids_2)

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = CompiledClip(context=Context(), import_to=import_to)

    module = CompiledModule.get_mlir_module(inst)

    model_metadata_encode = {
        "model_name": hf_model_name + "_text_encoder",
        "input_shapes": [str((1, max_length)) for i in range(4)],
        "input_dtypes": ["int64" for i in range(4)],
        "use_attention_mask": False,
    }
    module = AddMetadataPass(module, model_metadata_encode, "encode_prompts").run()
    module_str = str(module)

    if compile_to != "vmfb":
        return module_str, tokenizers
    else:
        vmfb_path = utils.compile_to_vmfb(
            module_str,
            device,
            target,
            ireec_flags,
            safe_name,
            return_path=not exit_on_vmfb,
            const_expr_hoisting=True,
            attn_spec=attn_spec,
        )
        return module_str, vmfb_path


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    mod_str, _ = export_prompt_encoder(
        args.hf_model_name,
        args.hf_auth_token,
        args.max_length,
        args.batch_size,
        args.precision,
        args.compile_to,
        args.external_weights,
        args.external_weight_path,
        args.device,
        args.iree_target_triple,
        args.ireec_flags + args.clip_flags,
        exit_on_vmfb=True,
        pipeline_dir=args.pipeline_dir,
        input_mlir=args.input_mlir,
        attn_spec=args.attn_spec,
    )
    if args.input_mlir:
        exit()
    safe_name_1 = safe_name = utils.create_safe_name(
        args.hf_model_name, f"{str(args.max_length)}_{args.precision}_prompt_encoder"
    )
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
