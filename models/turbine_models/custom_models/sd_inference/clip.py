# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re

from iree.compiler.ir import Context
from iree.turbine.aot import *
from iree.turbine.transforms.general.add_metadata import AddMetadataPass
from turbine_models.custom_models.sd_inference import utils
import torch
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor
from turbine_models.turbine_tank import turbine_tank


@torch.no_grad()
def export_clip_model(
    hf_model_name,
    batch_size: int = 1,
    max_length: int = 64,
    precision: str = "fp16",
    compile_to: str = "torch",
    external_weights: str = None,
    external_weight_path: str = None,
    device: str = "llvm-cpu",
    target: str = "x86_64-linux-gnu",
    ireec_flags: str = None,
    exit_on_vmfb: bool = False,
    pipeline_dir: str = None,
    input_mlir: str = None,
    attn_spec: str = None,
    weights_only: bool = False,
    upload_ir: bool = False,
    decomp_attn: bool = True,
):
    input_len = max_length
    safe_name = utils.create_safe_name(
        hf_model_name, f"_bs{batch_size}_{str(max_length)}-{precision}-clip"
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

    decomp_list = []
    if decomp_attn == True:
        decomp_list = [
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.scaled_dot_product_attention,
        ]
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=decomp_list,
    ):
        if "google/t5" in hf_model_name:
            from transformers import T5Tokenizer, T5Model

            tokenizer = T5Tokenizer.from_pretrained(hf_model_name)
            text_encoder_model = T5Model.from_pretrained(hf_model_name)
            input_len = 512

        else:
            # TODO: Add better filtering mechanism for things that require CLIPProcessor
            if "openai" in hf_model_name:
                tokenizer = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-large-patch14"
                )
                hf_subfolder = ""  # CLIPProcessor does not have a subfolder
                input_len = 10
            else:
                # Load the tokenizer and text encoder to tokenize and encode the text.
                tokenizer = CLIPTokenizer.from_pretrained(
                    hf_model_name,
                    subfolder="tokenizer",
                )
                hf_subfolder = "text_encoder"

            text_encoder_model = CLIPTextModel.from_pretrained(
                hf_model_name,
                subfolder=hf_subfolder,
            )
        if precision == "fp16":
            text_encoder_model = text_encoder_model.half()
        mapper = {}
        utils.save_external_weights(
            mapper, text_encoder_model, external_weights, external_weight_path
        )
        if weights_only:
            return external_weight_path

        if "google/t5" in hf_model_name:
            input_shapes = [(batch_size, input_len), (batch_size, input_len)]

            class CompiledTextEncoder(CompiledModule):
                if external_weights:
                    params = export_parameters(
                        text_encoder_model,
                        external=True,
                        external_scope="",
                        name_mapper=mapper.get,
                    )
                else:
                    params = export_parameters(text_encoder_model)

                def encode_tokens(
                    self,
                    inp=AbstractTensor(1, input_len, dtype=torch.int64),
                    decoder_input_ids=AbstractTensor(1, input_len, dtype=torch.int64),
                ):
                    return jittable(text_encoder_model.forward)(
                        input_ids=inp, decoder_input_ids=decoder_input_ids
                    )

        else:
            input_shapes = [str((batch_size, input_len)), str((batch_size, input_len))]

            class CompiledTextEncoder(CompiledModule):
                if external_weights:
                    params = export_parameters(
                        text_encoder_model,
                        external=True,
                        external_scope="",
                        name_mapper=mapper.get,
                    )
                else:
                    params = export_parameters(text_encoder_model)

                def encode_tokens_attn_mask(
                    self,
                    inp=AbstractTensor(1, input_len, dtype=torch.int64),
                    attn_mask=AbstractTensor(1, input_len, dtype=torch.int64),
                ):
                    return jittable(text_encoder_model.forward)(
                        input_ids=inp, attention_mask=attn_mask
                    )

                def encode_tokens(
                    self,
                    inp=AbstractTensor(1, input_len, dtype=torch.int64),
                ):
                    return jittable(text_encoder_model.forward)(input_ids=inp)

        import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
        inst = CompiledTextEncoder(context=Context(), import_to=import_to)
        module = CompiledModule.get_mlir_module(inst)

    model_metadata_attn_mask = {
        "model_name": hf_model_name + "_text_encoder",
        "input_shapes": input_shapes,
        "input_dtypes": ["int64", "int64"],
        "use_attention_mask": True,
    }
    model_metadata_encode = {
        "model_name": hf_model_name + "_text_encoder",
        "input_shapes": input_shapes[0],
        "input_dtypes": ["int64"],
        "use_attention_mask": False,
    }
    module = AddMetadataPass(
        module, model_metadata_attn_mask, "encode_tokens_attn_mask"
    ).run()
    module = AddMetadataPass(module, model_metadata_encode, "encode_tokens").run()

    module_str = str(module)
    if compile_to != "vmfb":
        return module_str
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
        return vmfb_path


if __name__ == "__main__":
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

    mod_str, _ = export_clip_model(
        args.hf_model_name,
        args.max_length,
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
        weights_only=False,
        upload_ir=False,
    )
    if args.input_mlir:
        exit()
    safe_name = utils.create_safe_name(
        args.hf_model_name, f"{str(args.max_length)}_{args.precision}_clip"
    )
    with open(f"{safe_name}.mlir", "w+") as f:
        f.write(mod_str)
    print("Saved to", safe_name + ".mlir")
