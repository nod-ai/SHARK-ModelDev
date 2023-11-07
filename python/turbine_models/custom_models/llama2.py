from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import torch
from torch.utils import _pytree as pytree
from shark_turbine.aot import *
from iree.compiler.ir import Context
import os

#I hate this but its faster than generating it from scratch each time you want to compile this.
json_schema = """
[1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}]}]
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
parser.add_argument("--compile_to", type=str, help="torch, linalg")
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="meta-llama/Llama-2-7b-chat-hf",
)


prompt = "The quick brown fox jumps over the lazy dog."


def slice_up_to_step(global_pkv, seq_step, heads, hidden_dim):
    all_pkv_tensors = []
    for i in range(heads * 2):
        sliced = IREE.tensor_slice(
            global_pkv, i, 0, (0, heads), (0, seq_step), (0, hidden_dim)
        )  # sequence context dim
        all_pkv_tensors.append(
            IREE.tensor_reshape(sliced, 1, heads, seq_step, hidden_dim)
        )

    return all_pkv_tensors


def update_state(state, state_updates, seq_step, heads, hidden_dim):
    all_updates = []
    for i in range(heads * 2):
        update = IREE.tensor_reshape(
            state_updates[i], 1, 1, heads, 1, hidden_dim
        )
        all_updates.append(
            IREE.tensor_update(state, update, i, 0, 0, seq_step, 0)
        )
    return all_updates


def get_llama_ir(
     hf_model_name, hf_auth_token, compile_to,
):
    state_schema = pytree.treespec_loads(json_schema)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
        use_auth_token=hf_auth_token,
    )
    mod = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        torch_dtype=torch.float,
        use_auth_token=hf_auth_token,
    )
    initial_input = tokenizer(prompt, return_tensors="pt")
    # TODO: generate these values instead of magic numbers
    MAX_STEP_SEQ = 4095
    HEADS = 32
    HIDDEN_DIM = 128
    BATCH_SIZE = 1
    global_pkv = torch.zeros(
        size=(HEADS * 2, BATCH_SIZE, HEADS, MAX_STEP_SEQ, HIDDEN_DIM),
        dtype=torch.float32,
    )
    seq_step = AbstractIndex

    class StateUpdateModule(CompiledModule):
        params = export_parameters(mod, initialize=False)
        global_state = export_global(global_pkv, mutable=True, initialize=True)
        global_seq_step = export_global(
            seq_step, mutable=True, initialize=True
        )

        def run_initialize(
            self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)
        ):
            init_const = [x.dynamic_dim(1) < MAX_STEP_SEQ]
            token, *state = self.initialize(x, constraints=init_const)
            updates = []
            self.global_seq_step = IREE.tensor_dim(
                state[0], 2 
            )  # 2nd dimension of arbitrarily 0th kv tensor
            for i in range(HEADS * 2):
                slice_of_state = IREE.tensor_reshape(
                    state[i], 1, 1, HEADS, self.global_seq_step, HIDDEN_DIM
                )
                updates.append(
                    IREE.tensor_update(
                        self.global_state, slice_of_state, i, 0, 0, 0, 0
                    )
                )
            return token

        def run_forward(self, x=AbstractTensor(1, None, dtype=torch.int64)):
            state_arg = slice_up_to_step(
                self.global_state, self.global_seq_step, HEADS, HIDDEN_DIM
            )
            forw_const = [state_arg[0].dynamic_dim(2) < MAX_STEP_SEQ] + [
                x.dynamic_dim(2) == (state_arg[0].dynamic_dim(2))
                for x in state_arg[1:]
            ]
            token, *state_update = self.forward(
                x, *state_arg, constraints=forw_const
            )
            self.global_seq_step = self.global_seq_step + 1
            res = update_state(
                self.global_state,
                state_update,
                self.global_seq_step,
                HEADS,
                HIDDEN_DIM,
            )

            return token

        def get_global_state(self):
            return self.global_state

        def get_seq_step(self):
            return self.global_seq_step

        @jittable
        def initialize(input_ids):
            result = mod.forward(input_ids)
            state1_flat, _ = pytree.tree_flatten(result.past_key_values)
            token1 = torch.argmax(result.logits[:, -1, :], dim=1)
            token1 = token1[None, :]
            return token1, *state1_flat

        @jittable
        def forward(token0: torch.Tensor, *state0_flat):
            # Unpad the states.
            state0 = pytree.tree_unflatten(state0_flat, state_schema)
            result = mod.forward(token0, past_key_values=state0)
            state1_flat, _ = pytree.tree_flatten(result.past_key_values)
            state1_flat = [x[:, :, -1:, :] for x in state1_flat]
            token1 = torch.argmax(result.logits[:, -1, :], dim=1)
            token1 = token1[None, :]
            return token1, *state1_flat

    import_to = "IMPORT" if compile_to == "torch" else "INPUT"
    inst = StateUpdateModule(context=Context(), import_to=import_to)
    module_str = str(CompiledModule.get_mlir_module(inst))
    return module_str, tokenizer


if __name__ == "__main__":
    args = parser.parse_args()
    module_str = get_llama_ir(args.hf_model_name, args.hf_auth_token, args.compile_to)
    safe_name = args.hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    dialect_postfix = args.compile_to
    with open(f"{safe_name}_{args.compile_to}.mlir", "w+") as f:
        f.write(module_str)
    