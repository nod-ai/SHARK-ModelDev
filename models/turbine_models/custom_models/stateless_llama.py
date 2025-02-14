import os
import sys
import re
import json
from turbine_models.turbine_tank import turbine_tank
from pathlib import Path

os.environ["TORCH_LOGS"] = "dynamic"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils import _pytree as pytree
from iree.turbine.aot import *
from iree.compiler.ir import Context
from turbine_models.custom_models.llm_optimizations.streaming_llm.modify_llama import (
    enable_llama_pos_shift_attention,
)
from turbine_models.custom_models.sd_inference.utils import compile_to_vmfb
from turbine_models.model_runner import vmfbRunner

from turbine_models.custom_models import remap_gguf
import safetensors

BATCH_SIZE = 1

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="Trelis/Llama-2-7b-chat-hf-function-calling-v2",
)
parser.add_argument("--quantization", type=str, default="unquantized")
parser.add_argument("--external_weight_file", type=str, default="")
parser.add_argument(
    "--vmfb_path", type=str, default=None, help="Path/name to store compiled vmfb."
)
parser.add_argument(
    "--external_weights",
    type=str,
    default=None,
    help="saves ir/vmfb without global weights for size and readability, options [gguf, safetensors]",
)
parser.add_argument(
    "--precision", type=str, default="fp16", help="dtype of model [f16, f32]"
)
parser.add_argument(
    "--device", type=str, default="llvm-cpu", help="llvm-cpu, cuda, vulkan, rocm"
)
# TODO: Bring in detection for target triple
parser.add_argument(
    "--iree_target_triple",
    type=str,
    default="host",
    help="Specify vulkan target triple or rocm/cuda target device.",
)
parser.add_argument("--vulkan_max_allocation", type=str, default="4294967296")
parser.add_argument(
    "--streaming_llm",
    action="store_true",
    help="Compile LLM with StreamingLLM optimizations",
)
parser.add_argument(
    "--decomp_attn",
    action="store_true",
    help="Decompose attention ops at fx graph level.",
)


def generate_schema(num_layers):
    null = None
    schema = [1, {"type": "builtins.tuple", "context": "null", "children_spec": []}]
    kv_schema_per_layer = {
        "type": "builtins.tuple",
        "context": "null",
        "children_spec": [
            {"type": null, "context": null, "children_spec": []},
            {"type": null, "context": null, "children_spec": []},
        ],
    }
    for i in range(num_layers):
        schema[1]["children_spec"].append(kv_schema_per_layer)
    return json.dumps(schema)


def slice_up_to_step(k_caches, v_caches, seq_step, heads, hidden_dim, num_layers):
    all_pkv_tensors = []
    for i in range(num_layers * 2):
        # Numpy semantic: sliced = global_pkv[i, 0, 0:seq_step, 0:heads, 0:hidden_dim]
        # Generates tensor<1 x 1 x seq_step x heads x hidden_dim>
        if i % 2 == 0:
            sliced = IREE.tensor_slice(
                k_caches["layer_idx"][i // 2],
                0,
                (0, seq_step),
                (0, heads),
                (0, hidden_dim),
            )  # sequence context dim
        else:
            sliced = IREE.tensor_slice(
                v_caches["layer_idx"][i // 2],
                0,
                (0, seq_step),
                (0, heads),
                (0, hidden_dim),
            )  # sequence context dim
        all_pkv_tensors.append(
            IREE.tensor_reshape(sliced, 1, seq_step, heads, hidden_dim)
        )
    return all_pkv_tensors


def export_transformer_model(
    hf_model_name,
    hf_auth_token=None,
    compile_to="torch",
    external_weights=None,
    external_weight_file=None,
    quantization=None,
    precision=None,
    device=None,
    target_triple="x86_64-unknown-linux-gnu",
    vulkan_max_allocation=None,
    streaming_llm=False,
    vmfb_path=None,
    upload_ir=False,
    mod=None,
    tokenizer=None,
    decomp_attn=False,
    input_mlir=None,
    iree_flags=[],
):
    safe_name = hf_model_name.replace("-", "_").replace("/", "_")
    if streaming_llm:
        safe_name += "_streaming"
    if not vmfb_path:
        vmfb_path = safe_name + "_" + target_triple

    ukernel_supported_arch = {"gfx90a", "gfx940", "gfx1030", "gfx1100"}
    if target_triple in ukernel_supported_arch:
        iree_flags.extend(["--iree-rocm-enable-ukernels=argmax"])
    if input_mlir is not None:
        vmfb_path = compile_to_vmfb(
            input_mlir,
            device,
            target_triple,
            ireec_flags=iree_flags,
            safe_name=vmfb_path.split(".vmfb")[0],
            return_path=True,
            const_expr_hoisting=True,
            mlir_source="file",
            save_mlir=False,
            attn_spec="mfma" if "gfx9" in target_triple else "wmma",
        )
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            use_fast=False,
            token=hf_auth_token,
        )
    if mod == None:
        mod = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float,
            token=hf_auth_token,
        )
    schema_json = generate_schema(mod.config.num_hidden_layers)
    state_schema = pytree.treespec_loads(schema_json)
    if streaming_llm:
        enable_llama_pos_shift_attention(mod)
    dtype = torch.float32
    if precision == "f16":
        mod = mod.half()
        dtype = torch.float16

    # TODO: generate these values instead of magic numbers
    NUM_LAYERS = mod.config.num_hidden_layers
    HEADS = getattr(mod.config, "num_key_value_heads", None)
    if HEADS is None:
        HEADS = mod.config.num_attention_heads
    HIDDEN_DIM = int(mod.config.hidden_size / mod.config.num_attention_heads)
    BATCH_SIZE = 1
    MAX_STEP_SEQ = mod.config.max_position_embeddings - 1
    global_pkv = torch.zeros(
        size=(BATCH_SIZE, MAX_STEP_SEQ, HEADS, HIDDEN_DIM),
        dtype=dtype,
    )
    kv_cache_structure = {
        "layer_idx": [abstractify(global_pkv) for _ in range(NUM_LAYERS)],
    }

    mapper = {}
    if external_weights is not None:
        if external_weights == "safetensors":
            mod_params = dict(mod.named_parameters())
            for name in mod_params:
                mapper["params." + name] = name
            if external_weight_file:
                if os.path.exists(external_weight_file) == False:
                    safetensors.torch.save_file(mod_params, external_weight_file)

        elif external_weights == "gguf":
            tensor_mapper = remap_gguf.TensorNameMap(remap_gguf.MODEL_ARCH.LLAMA, HEADS)
            mapper = tensor_mapper.mapping

    initial_table = decompositions.current_aot_decompositions()
    print("Decomposing torch SDPA")
    with decompositions.extend_aot_decompositions(
        from_current=True,
        add_ops=[
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten.masked_fill_.Scalar,
            torch.ops.aten.copy,
        ],
    ):
        current_table = decompositions.current_aot_decompositions()

        class StateUpdateModule(CompiledModule):
            if external_weights:
                params = export_parameters(
                    mod, external=True, external_scope="", name_mapper=mapper.get
                )
            else:
                params = export_parameters(mod)
            global_seq_step = export_global(AbstractIndex, mutable=True)
            global_k_caches = export_global_tree(
                kv_cache_structure, uninitialized=True, mutable=True
            )
            global_v_caches = export_global_tree(
                kv_cache_structure, uninitialized=True, mutable=True
            )

            def run_initialize(
                self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)
            ):
                dynamic_shapes_init = {
                    "arg0_1": {1: torch.export.Dim("dim", max=MAX_STEP_SEQ - 1)}
                }
                token, *state = self.initialize(x, dynamic_shapes=dynamic_shapes_init)
                self.global_seq_step = IREE.tensor_dim(
                    state[0], 1
                )  # ? dimension of arbitrarily 0th kv tensor
                for i in range(NUM_LAYERS):
                    slice_of_state = IREE.tensor_reshape(
                        state[i * 2], 1, self.global_seq_step, HEADS, HIDDEN_DIM
                    )
                    self.global_k_caches["layer_idx"][i] = IREE.tensor_update(
                        self.global_k_caches["layer_idx"][i], slice_of_state, 0, 0, 0, 0
                    )
                for i in range(NUM_LAYERS):
                    slice_of_state = IREE.tensor_reshape(
                        state[i * 2 + 1], 1, self.global_seq_step, HEADS, HIDDEN_DIM
                    )
                    self.global_v_caches["layer_idx"][i] = IREE.tensor_update(
                        self.global_v_caches["layer_idx"][i], slice_of_state, 0, 0, 0, 0
                    )
                return token

            def run_forward(self, x=AbstractTensor(1, 1, dtype=torch.int64)):
                state_arg = slice_up_to_step(
                    self.global_k_caches,
                    self.global_v_caches,
                    self.global_seq_step,
                    HEADS,
                    HIDDEN_DIM,
                    NUM_LAYERS,
                )
                state_arg0_dim = torch.export.Dim(
                    "state_arg0_dim", max=MAX_STEP_SEQ - 1
                )
                dynamic_shapes_forw = {"arg0_1": None, "arg1_1": {1: state_arg0_dim}}
                for state_arg_idx in range(2, len(state_arg) + 1):
                    current_dim_dict = {f"arg{state_arg_idx}_1": {1: state_arg0_dim}}
                    dynamic_shapes_forw = {**dynamic_shapes_forw, **current_dim_dict}
                token, *state_update = self.forward(
                    x, *state_arg, dynamic_shapes=dynamic_shapes_forw
                )
                for i in range(NUM_LAYERS):
                    update = IREE.tensor_reshape(
                        state_update[i * 2], 1, 1, HEADS, HIDDEN_DIM
                    )
                    self.global_k_caches["layer_idx"][i] = IREE.tensor_update(
                        self.global_k_caches["layer_idx"][i],
                        update,
                        0,
                        self.global_seq_step,
                        0,
                        0,
                    )
                for i in range(NUM_LAYERS):
                    update = IREE.tensor_reshape(
                        state_update[i * 2 + 1], 1, 1, HEADS, HIDDEN_DIM
                    )
                    self.global_v_caches["layer_idx"][i] = IREE.tensor_update(
                        self.global_v_caches["layer_idx"][i],
                        update,
                        0,
                        self.global_seq_step,
                        0,
                        0,
                    )
                self.global_seq_step = self.global_seq_step + 1
                return token

            def get_seq_step(self):
                return self.global_seq_step

            @jittable
            def initialize(input_ids):
                result = mod.forward(input_ids)
                state1_flat, _ = pytree.tree_flatten(result.past_key_values)
                token1 = torch.argmax(result.logits[:, -1, :], dim=1)
                token1 = token1[None, :]
                state1_flat = [torch.transpose(x, 1, 2) for x in state1_flat]
                return token1, *state1_flat

            @jittable
            def forward(token0: torch.Tensor, *state0_flat):
                # Unpad the states.
                state0_flat = [torch.transpose(x, 1, 2) for x in state0_flat]
                state0 = pytree.tree_unflatten(state0_flat, state_schema)
                result = mod.forward(token0, past_key_values=state0)
                state1_flat, _ = pytree.tree_flatten(result.past_key_values)
                state1_flat = [
                    torch.transpose(x[:, :, -1:, :], 1, 2) for x in state1_flat
                ]
                token1 = torch.argmax(result.logits[:, -1, :], dim=1)
                token1 = token1[None, :]
                return token1, *state1_flat

        class StreamingStateUpdateModule(StateUpdateModule):
            def run_cached_initialize(
                self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)
            ):
                state_arg = slice_up_to_step(
                    self.global_k_caches,
                    self.global_v_caches,
                    self.global_seq_step,
                    HEADS,
                    HIDDEN_DIM,
                    NUM_LAYERS,
                )
                state_arg0_dim1 = torch.export.Dim(
                    "state_arg0_dim1", max=MAX_STEP_SEQ - 1
                )
                x_dim = torch.export.Dim("x_dim", max=MAX_STEP_SEQ - 1)
                dynamic_shapes_forw = {
                    "arg0_1": {1: x_dim},
                    "arg1_1": {1: state_arg0_dim1},
                }
                for state_arg_idx in range(2, len(state_arg) + 1):
                    current_dim_dict = {f"arg{state_arg_idx}_1": {1: state_arg0_dim1}}
                    dynamic_shapes_forw = {**dynamic_shapes_forw, **current_dim_dict}
                token, *state = self.cached_initialize(
                    x, *state_arg, dynamic_shapes=dynamic_shapes_forw
                )
                len_of_new_tokens = IREE.tensor_dim(
                    state[0], 1
                )  # ? dimension of arbitrarily 0th kv tensor
                for i in range(NUM_LAYERS):
                    slice_of_state = IREE.tensor_reshape(
                        state[i * 2], 1, len_of_new_tokens, HEADS, HIDDEN_DIM
                    )
                    self.global_k_caches["layer_idx"][i] = IREE.tensor_update(
                        self.global_k_caches["layer_idx"][i],
                        slice_of_state,
                        0,
                        self.global_seq_step,
                        0,
                        0,
                    )
                for i in range(NUM_LAYERS):
                    slice_of_state = IREE.tensor_reshape(
                        state[i * 2 + 1], 1, len_of_new_tokens, HEADS, HIDDEN_DIM
                    )
                    self.global_v_caches["layer_idx"][i] = IREE.tensor_update(
                        self.global_v_caches["layer_idx"][i],
                        slice_of_state,
                        0,
                        self.global_seq_step,
                        0,
                        0,
                    )
                self.global_seq_step = self.global_seq_step + len_of_new_tokens
                return token

            @jittable
            def cached_initialize(input_ids, *state0_flat):
                # Unpad the states.
                cur_token_len = state0_flat[0].size(1)
                state0_flat = [torch.transpose(x, 1, 2) for x in state0_flat]
                state0 = pytree.tree_unflatten(state0_flat, state_schema)
                result = mod.forward(input_ids, past_key_values=state0)
                state1_flat, _ = pytree.tree_flatten(result.past_key_values)
                state1_flat = [
                    torch.transpose(x[:, :, cur_token_len:, :], 1, 2)
                    for x in state1_flat
                ]
                token1 = torch.argmax(result.logits[:, -1, :], dim=1)
                token1 = token1[None, :]
                return token1, *state1_flat

            # Streaming-LLM KVCache evict algorithm:
            # slice1 = KVCache[0 : sink]
            # slice2 = KVCache[seq_len - window_size : seq_len]
            # KVCache = torch.cat([slice1, slice2])
            # TODO: Add move to handle overlap of data.
            def evict_kvcache_space(self):
                # TODO: Replace hardcoded with global variable.
                sink_size = 4
                window_size = 252
                most_recent_window = self.global_seq_step + (-window_size)
                for i in range(NUM_LAYERS):
                    update_window_state = IREE.tensor_slice(
                        self.global_k_caches["layer_idx"][i],
                        0,
                        (most_recent_window, window_size),
                        (0, HEADS),
                        (0, HIDDEN_DIM),
                    )  # sequence context dim
                    self.global_k_caches["layer_idx"][i] = IREE.tensor_update(
                        self.global_k_caches["layer_idx"][i],
                        update_window_state,
                        0,
                        sink_size,
                        0,
                        0,
                    )
                for i in range(NUM_LAYERS):
                    update_window_state = IREE.tensor_slice(
                        self.global_v_caches["layer_idx"][i],
                        0,
                        (most_recent_window, window_size),
                        (0, HEADS),
                        (0, HIDDEN_DIM),
                    )  # sequence context dim
                    self.global_v_caches["layer_idx"][i] = IREE.tensor_update(
                        self.global_v_caches["layer_idx"][i],
                        update_window_state,
                        0,
                        sink_size,
                        0,
                        0,
                    )
                self.global_seq_step.set(window_size + sink_size)
                return self.global_seq_step

        import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
        if streaming_llm:
            print("Compiling with Streaming LLM")
            inst = StreamingStateUpdateModule(context=Context(), import_to=import_to)
        else:
            inst = StateUpdateModule(context=Context(), import_to=import_to)
        # TODO: Integrate with external parameters to actually be able to run
        # TODO: Make more generalizable to be able to quantize with all  compile_to options
        if quantization == "int4" and not compile_to == "linalg":
            from iree.turbine.transforms.quantization import mm_group_quant

            mm_group_quant.MMGroupQuantRewriterPass(
                CompiledModule.get_mlir_module(inst).operation
            ).run()
        module_str = str(CompiledModule.get_mlir_module(inst))
    if upload_ir:
        with open(f"{safe_name}.mlir", "w+") as f:
            f.write(module_str)
        model_name_upload = hf_model_name.replace("/", "_")
        blob_name = turbine_tank.uploadToBlobStorage(
            str(os.path.abspath(f"{safe_name}.mlir")),
            f"{model_name_upload}/{model_name_upload}.mlir",
        )
    if compile_to != "vmfb":
        return module_str, tokenizer
    else:
        blob_name = compile_to_vmfb(
            module_str,
            device,
            target_triple,
            ireec_flags=iree_flags,
            safe_name=vmfb_path.split(".vmfb")[0],
            return_path=True,
            const_expr_hoisting=True,
            mlir_source="str",
            save_mlir=False,
            attn_spec="mfma" if "gfx9" in target_triple else "wmma",
        )
        if upload_ir:
            return blob_name
        return blob_name, tokenizer


llm_model_map = {
    "meta-llama/Llama-2-7b-chat-hf": {
        "initializer": export_transformer_model,
        "hf_model_name": "meta-llama/Llama-2-7b-chat-hf",
        "compile_flags": ["--iree-opt-const-expr-hoisting=False"],
        "stop_token": 2,
        "max_tokens": 4096,
        "system_prompt": """<s>[INST] <<SYS>>Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>""",
    },
    "Trelis/Llama-2-7b-chat-hf-function-calling-v2": {
        "initializer": export_transformer_model,
        "hf_model_name": "Trelis/Llama-2-7b-chat-hf-function-calling-v2",
        "compile_flags": ["--iree-opt-const-expr-hoisting=False"],
        "stop_token": 2,
        "max_tokens": 4096,
        "system_prompt": """<s>[INST] <<SYS>>Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>""",
    },
    "TinyPixel/small-llama2": {
        "initializer": export_transformer_model,
        "hf_model_name": "TinyPixel/small-llama2",
        "compile_flags": ["--iree-opt-const-expr-hoisting=True"],
        "stop_token": 2,
        "max_tokens": 1024,
        "system_prompt": """<s>[INST] <<SYS>>Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>""",
    },
}

    # args.hf_model_name,
    # args.scheduler_id,
    # args.precision,
    # args.device
    # args.iree_target_triple,
    # flags,
    # args.pipeline_dir,
    # args.external_weights_dir,
    # args.external_weights,
    # args.hf_auth_token,


class StatelessLlama:
    def __init__(
        self,
        hf_model_name: str,
        scheduler_id: str,
        precision: str,
        device: str,
        iree_target_triple: str,
        ireec_flags: list = [],
        pipeline_dir: str | Path = "./shark_vmfbs",
        external_weights_dir: str | Path = "./shark_weights",
        external_weights: str = "safetensors",
        hf_auth_token: str = None,
    ):
        self.hf_model_name = hf_model_name
        self.iree_dtype = "float32" if precision == "fp32" else "float16"
        self.torch_dtype = torch.float32 if precision == "fp32" else torch.float16
        self.precision = precision
        self.device = device
        self.iree_target_triple = iree_target_triple
        self.ireec_flags = ireec_flags
        self.pipeline_dir = pipeline_dir
        self.external_weights_dir = external_weights_dir
        self.external_weights = external_weights

        self.first_input = True
        self.max_tokens = llm_model_map[self.hf_model_name]["max_tokens"]
        self.global_iter = 0
        self.prev_token_len = 0
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model_name,
            use_fast=False,
            use_auth_token=hf_auth_token,
        )
        self.safe_name = "_".join(
            [
                self.hf_model_name.replace("/", "_").replace("-", "_"),
                self.precision,
            ]
        )
        self.model = None
        self.hf_auth_token=hf_auth_token

    # FILE MANAGEMENT AND PIPELINE SETUP

    def check_prepared(
        self,
        mlir: str,
        vmfb: str,
        weight: str,
        interactive: bool = False,
        quantization: str = None,
    ):
        ready, vmfb, weight = self.is_prepared(vmfb, weight)
        if not ready:
            if interactive:
                do_continue = input(
                    f"\nIt seems you are missing some necessary files. Would you like to generate them now? (y/n)"
                )
                if do_continue.lower() != "y":
                    exit()
            else:
                do_continue = "y"
            if do_continue.lower() == "y":
                if vmfb is None:
                    v, w = self.export(input_mlir=mlir, quantization=quantization)
                    vmfb = v
                    if weight is None:
                        weight = w
                if weight is None:
                    _, w = self.export(weights_only=True, quantization=quantization)
                    weight = w
                ready, vmfb, weight = self.is_prepared(vmfb, weight)
                if ready:
                    print("All necessary files found.")
                    return vmfb, weight
                else:
                    print("There was an error generating the necessary files.")
                    exit()
        else:
            print("All necessary files found. Loading pipeline.")
        return vmfb, weight

    def is_prepared(self, vmfb, weight):
        missing = []
        default_filepath = os.path.join(self.pipeline_dir, self.safe_name + ".vmfb")

        # vmfb
        if vmfb is None and os.path.exists(default_filepath):
            vmfb = default_filepath
        else:
            missing.append(vmfb)

        # External weight
        if not (weight is not None and os.path.exists(weight)):            
            if self.external_weights is None:
                weight = None
        else:
            default_name = os.path.join(
                self.external_weights_dir, self.safe_name + "." + self.external_weights
            )
            if weight is None and os.path.exists(default_name):
                weight = os.path.join(default_name)
            else:
                missing.append(weight)
        if len(missing) > 0:
            # print(f"Missing files: " + ", ".join(missing))
            return False, vmfb, weight
        else:
            return True, vmfb, weight

    # IMPORT / COMPILE PHASE

    def export(
        self,
        quantization: str = None,
        input_mlir: str = None,
        weights_only: bool = False,
    ):
        safe_name = self.hf_model_name.replace("-", "_").replace("/", "_")
        # if self.streaming_llm:
        safe_name += "_streaming"

        if not os.path.exists(self.pipeline_dir):
            os.makedirs(self.pipeline_dir)
        if self.external_weights_dir:
            if not os.path.exists(self.external_weights_dir):
                os.makedirs(external_weights_dir, exist_ok=True)
            external_weight_path = os.path.join(
                self.external_weights_dir, safe_name + self.external_weights
            )
        elif self.external_weights is None:
            print(
                "No external weights type specified using --external_weights, weights for imported .mlir files will not be externalized."
            )
            external_weight_path = None
        else:
            print(
                f"No external weights directory specified using --external_weights_dir, we assume you have your own weights in {self.pipeline_dir}."
            )
            external_weights_dir = self.pipeline_dir
            external_weight_path = os.path.join(
                self.pipeline_dir, safe_name + self.external_weights
            )
        if weights_only:
            input_mlir = None

        _, vmfb = export_transformer_model(
            self.hf_model_name,
            hf_auth_token=self.hf_auth_token,
            compile_to="vmfb",
            external_weights=self.external_weights,
            external_weight_file=external_weight_path,
            quantization=quantization,
            precision=self.precision,
            device=self.device,
            target_triple=self.iree_target_triple,
            vulkan_max_allocation=None,
            streaming_llm=True,
            vmfb_path=os.path.join(self.pipeline_dir, safe_name + ".vmfb"),
            upload_ir=False,
            mod=None,
            tokenizer=None,
            decomp_attn=False,
            input_mlir=input_mlir,
            iree_flags=self.ireec_flags,
        )
        return vmfb, external_weight_path

    # LOAD

    def load_pipeline(
        self,
        vmfb: str,
        weight: str,
        rt_device: str = "local-task",
        compiled_pipeline: bool = False,
    ):
        self.model = vmfbRunner(rt_device, vmfb, weight)

    # RUN

    def chat(self, prompt):
        prompt = self.sanitize_prompt(prompt)

        input_tensor = self.tokenizer(prompt, return_tensors="pt").input_ids

        def format_out(results):
            return torch.tensor(results.to_host()[0][0])

        history = []
        for iter in range(self.max_tokens):
            # if self.streaming_llm:
            token_slice = max(self.prev_token_len - 1, 0)
            input_tensor = input_tensor[:, token_slice:]
            # if self.streaming_llm and self.model["get_seq_step"]() > 600:
            if self.model["get_seq_step"]() > 600:
                print("Evicting cache space!")
                self.model["evict_kvcache_space"]()
            token_len = input_tensor.shape[-1]
            device_inputs = [
                ireert.asdevicearray(self.device, input_tensor)
            ]
            if self.first_input: # or not self.streaming_llm:
                st_time = time.time()
                token = self.model["run_initialize"](*device_inputs)
                total_time = time.time() - st_time
                token_len += 1
                self.first_input = False
            else:
                st_time = time.time()
                token = self.model["run_cached_initialize"](*device_inputs)
                total_time = time.time() - st_time
                token_len += 1

            history.append(format_out(token))
            while (
                format_out(token) != llm_model_map[self.hf_model_name]["stop_token"]
                and len(history) < self.max_tokens
            ):
                dec_time = time.time()
                if self.model["get_seq_step"]() > 600:
                    print("Evicting cache space!")
                    self.model["evict_kvcache_space"]()
                token = self.model["run_forward"](token)
                history.append(format_out(token))
                total_time = time.time() - dec_time
                yield self.tokenizer.decode(history), total_time

            self.prev_token_len = token_len + len(history)

            if format_out(token) == llm_model_map[self.hf_model_name]["stop_token"]:
                break

        for i in range(len(history)):
            if type(history[i]) != int:
                history[i] = int(history[i])
        result_output = self.tokenizer.decode(history)
        self.global_iter += 1
        return result_output, total_time

if __name__ == "__main__":
    from turbine_models.custom_models.llm_cmd_opts import args
    
    mlir = None #args.input_mlir
    vmfb = None
    weight = None

    flags = []
    if "cpu" in args.device:
        flags.extend(
            [
                "--iree-global-opt-enable-quantized-matmul-reassociation",
            ]
        )
    elif args.device == "vulkan":
        flags.extend(["--iree-stream-resource-max-allocation-size=4294967296"])
    elif args.device == "rocm":
        flags.extend(
            [
                "--iree-codegen-llvmgpu-enable-transform-dialect-jit=false",
                "--iree-llvmgpu-enable-prefetch=true",
                "--iree-opt-outer-dim-concat=true",
                "--iree-flow-enable-aggressive-fusion",
            ]
        )
        if "gfx9" in args.iree_target_triple:
            flags.extend(
                [
                    f"--iree-codegen-transform-dialect-library={get_mfma_spec_path(args.iree_target_triple, get_checkpoints_path())}",
                    "--iree-codegen-llvmgpu-use-vector-distribution=true",
                ]
            )
    flags.extend(llm_model_map[args.hf_model_name]["compile_flags"])

    if not args.pipeline_dir:
        args.pipeline_dir = "./shark_vmfbs"
    if not args.external_weights_dir and args.external_weights:
        args.external_weights_dir = args.pipeline_dir

    llama = StatelessLlama(
        args.hf_model_name,
        args.scheduler_id,
        args.precision,
        args.device,
        args.iree_target_triple,
        flags,
        args.pipeline_dir,
        args.external_weights_dir,
        args.external_weights,
        args.hf_auth_token,
    )
    vmfb, weight = llama.check_prepared(mlir, vmfb, weight, interactive=False, quantization="int4")
    llama.load_pipeline(vmfb, weight, args.rt_device, args.compiled_pipeline)
    llama.generate_images(
        args.prompt,
        args.negative_prompt,
        args.batch_count,
        args.guidance_scale,
        args.seed,
        False,
    )
