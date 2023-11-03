import os
import numpy as np
import re

os.environ["TORCH_LOGS"] = "dynamic"
from shark_turbine.dynamo.importer import FxImporter
from shark_turbine.dynamo.passes import turbine_cpu_pass_pipeline
import torch._dynamo as dynamo
from torch._export import dynamic_dim
from torch._export.constraints import constrain_as_size, constrain_as_value
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils import _pytree as pytree
import textwrap
from torch.fx import (
    GraphModule,
)
import collections
from torch._export.constraints import constrain_as_size, constrain_as_value
from shark_turbine.aot import *
from iree.compiler.ir import Context
from iree import runtime as ireert

BATCH_SIZE = 1
MAX_STEP_SEQ = 4095

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_vmfb", action="store_true")
parser.add_argument(
    "--hf_auth_token", type=str, help="The Hugging Face auth token, required"
)
parser.add_argument("--compile_to", type=str, help="torch, linalg, vmfb")
parser.add_argument(
    "--test",
    action="store_true",
    help="run stateless tests instead of exporting",
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="meta-llama/Llama-2-7b-chat-hf",
)
parser.add_argument("--schema_path", type=str, help="Schema path")

prompt = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
"""

class InferenceModel(torch.nn.Module):
    def __init__(
        self,
        args,
        base_model_name="meta-llama/Llama-2-7b-chat-hf",
        state_schema_path="examples/llama2_inference/llama2_state_schema.json",
    ):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float,
            use_auth_token=args.hf_auth_token,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=False,
            use_auth_token=args.hf_auth_token,
        )
        self.base_model_name = base_model_name
        if os.path.exists(state_schema_path):
            with open(state_schema_path, "r+") as f:
                self.state_schema = pytree.treespec_loads(f.read())
        else:
            self.generate_state_schema()

    def get_sample_input(self):
        initial_input = self.tokenizer(prompt, return_tensors="pt")
        return initial_input.input_ids

    def generate_state_schema(self):
        prompt = "hi"
        initial_input = self.tokenizer(prompt, return_tensors="pt")
        initial_results = self.base_model.forward(initial_input.input_ids)
        # Here we're fetching dims 1 and 3 from an existing pkv value from the base model,
        # and substituting the second dim with MAX_STEP_SEQ
        sample_shape = initial_results.past_key_values[0][0].shape
        pkv = pytree.tree_map(
            lambda x: torch.zeros(
                BATCH_SIZE,
                sample_shape[1],
                MAX_STEP_SEQ,
                sample_shape[3],
                dtype=x.dtype,
            ),
            initial_results.past_key_values,
        )
        _, self.state_schema = pytree.tree_flatten(pkv)

    def write_schema_to_file(self, schema_path=None):
        if schema_path == None:
            schema_path = (f"{self.model_name.split('/')[-1]}_schema.json",)
        print(f"Writing schema to: {schema_path}")
        with open(schema_path, "w+") as f:
            f.write(pytree.treespec_dumps(self.state_schema))

    def initialize(self, input_ids: torch.Tensor):
        result = self.base_model.forward(input_ids)
        state1_flat, _ = pytree.tree_flatten(result.past_key_values)
        token1 = torch.argmax(result.logits[:, -1, :], dim=1)
        token1 = token1[None, :]
        return token1, *state1_flat

    def forward(self, token0: torch.Tensor, *state0_flat):
        # Unpad the states.
        state0 = pytree.tree_unflatten(state0_flat, self.state_schema)
        result = self.base_model.forward(token0, past_key_values=state0)
        state1_flat, _ = pytree.tree_flatten(result.past_key_values)
        state1_flat = [x[:, :, -1:, :] for x in state1_flat]
        token1 = torch.argmax(result.logits[:, -1, :], dim=1)
        return token1, *state1_flat

    def compile(self, example_input_id, compile_dynamo=False):
        # Export initializer
        exp_initialize = dynamo.export(
            self.initialize,
            aten_graph=True,
            assume_static_by_default=True,
            constraints=[
                dynamic_dim(example_input_id, 1) < MAX_STEP_SEQ,
            ],
        )
        g_initialize, guards_initialize = exp_initialize(example_input_id)

        example_token, *example_state = self.initialize(example_input_id)

        # Export forward
        exp_forward = dynamo.export(
            self.forward,
            aten_graph=True,
            assume_static_by_default=True,
            # Constrain the first state dim and then form an equality
            # on all of the others. If we don't specify sufficient constraints
            # for these, Dynamo will print two pages of a copy-pastable version
            # of basically this based on what it found in the graph but wants
            # you to be explicit about.
            constraints=[dynamic_dim(example_state[0], 2) < MAX_STEP_SEQ]
            + [
                (dynamic_dim(x, 2) == (dynamic_dim(example_state[0], 2)))
                for x in example_state[1:]
            ],
        )
        g_forward, guards_forward = exp_forward(example_token, *example_state)
        if not compile_dynamo:
            return g_initialize, guards_initialize, g_forward, guards_forward
        g_initialize = import_compiler(g_initialize, [example_input_id])
        g_forward = import_compiler(g_forward, [example_token, *example_state])

        return g_initialize, guards_initialize, g_forward, guards_forward


def test_stateless_against_torch():
    def unpack_tensor(pkv, seq_step):
        return [pkv[i, :, :, : seq_step + 1, :] for i in range(64)]

    model = InferenceModel()

    def get_token_from_logits(logits):
        return torch.argmax(logits[:, -1, :], dim=1)

    input_ids = model.get_sample_input()
    example_token0, *state1_flat = model.initialize(input_ids)
    seq_step = state1_flat[0].shape[2]
    shape_default = list(state1_flat[0].shape)
    shape_default[2] = MAX_STEP_SEQ
    shape_default = tuple([64] + shape_default)
    session_state = torch.zeros(shape_default, dtype=state1_flat[0].dtype)
    for i in range(64):
        session_state[i, :, :, :seq_step, :] = state1_flat[i]
    # get base model results
    base_model_results = model.base_model.forward(input_ids)
    base_model_token = get_token_from_logits(base_model_results.logits)
    assert example_token0 == base_model_token
    base_model_pkv = base_model_results.past_key_values
    token = example_token0
    for i in range(15):
        next_input_token = torch.reshape(token, [1, 1])
        base_results = model.base_model.forward(
            next_input_token, past_key_values=base_model_pkv
        )
        base_token = get_token_from_logits(base_results.logits)
        base_model_pkv = base_results.past_key_values

        state_as_list = unpack_tensor(session_state, seq_step)
        token, *state1_flat_update = model.forward(
            next_input_token, *unpack_tensor(session_state, seq_step)
        )
        for i in range(64):
            session_state[
                i, :, :, seq_step : seq_step + 1, :
            ] = state1_flat_update[i]
        seq_step += 1

        print(f"stateless_token {model.tokenizer.decode(token)}")
        print(f"base_token {model.tokenizer.decode(base_token)}")
        print()
        assert token == base_token


def slice_up_to_step(global_pkv, seq_step, heads, hidden_dim):
    all_pkv_tensors = []
    for i in range(heads * 2):
        sliced = IREE.tensor_slice(
            global_pkv, i, 0, (0, seq_step), (0, heads), (0, hidden_dim)
        )  # sequence context dim
        all_pkv_tensors.append(
            IREE.tensor_reshape(sliced, 1, seq_step, heads, hidden_dim)
        )

    return all_pkv_tensors


def export_transformer_model(
    state_schema_path, hf_model_name, hf_auth_token, compile_to
):
    state_schema = None
    if state_schema_path == None:
        state_schema_path = (
            "examples/llama2_inference/llama2_state_schema.json"
        )
    if os.path.exists(state_schema_path):
        with open(state_schema_path, "r+") as f:
            state_schema = pytree.treespec_loads(f.read())

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
    example_input_id = initial_input.input_ids
    # TODO: generate these values instead of magic numbers
    HEADS = 32
    HIDDEN_DIM = 128
    BATCH_SIZE = 1
    global_pkv = torch.zeros(
        size=(HEADS * 2, BATCH_SIZE, MAX_STEP_SEQ, HEADS, HIDDEN_DIM),
        dtype=torch.float32,
    )
    seq_step = AbstractIndex

    class StateUpdateModule(CompiledModule):
        params = export_parameters(mod, initialize=False)
        global_state = export_global(global_pkv, mutable=True, initialize=False)
        global_seq_step = export_global(
            seq_step, mutable=True, initialize=False
        )

        def run_initialize(
            self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)
        ):
            init_const = [x.dynamic_dim(1) < MAX_STEP_SEQ]
            token, *state = self.initialize(x, constraints=init_const)
            updates = []
            self.global_seq_step = IREE.tensor_dim(
                state[0], 1
            )  # ? dimension of arbitrarily 0th kv tensor
            for i in range(HEADS * 2):
                slice_of_state = IREE.tensor_reshape(
                    state[i], 1, 1, self.global_seq_step, HEADS, HIDDEN_DIM
                )
                self.global_state = IREE.tensor_update(
                    self.global_state, slice_of_state, i, 0, 0, 0, 0
                )
            return token

        def run_forward(self, x=AbstractTensor(1, None, dtype=torch.int64)):
            state_arg = slice_up_to_step(
                self.global_state, self.global_seq_step, HEADS, HIDDEN_DIM
            )
            forw_const = [state_arg[0].dynamic_dim(1) < MAX_STEP_SEQ] + [
                x.dynamic_dim(1) == (state_arg[0].dynamic_dim(1))
                for x in state_arg[1:]
            ]
            token, *state_update = self.forward(
                x, *state_arg, constraints=forw_const
            )
            for i in range(HEADS * 2):
                update = IREE.tensor_reshape(
                    state_update[i], 1, 1, 1, HEADS, HIDDEN_DIM
                )
                self.global_state = IREE.tensor_update(
                    self.global_state, update, i, 0, self.global_seq_step, 0, 0
                )

            self.global_seq_step = self.global_seq_step + 1
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

    import_to = "IMPORT" if compile_to == "torch" else "INPUT"
    inst = StateUpdateModule(context=Context(), import_to=import_to)
    module_str = str(CompiledModule.get_mlir_module(inst))
    safe_name = hf_model_name.split("/")[-1].strip()
    safe_name = re.sub("-", "_", safe_name)
    if compile_to != "vmfb":
        dialect_postfix = compile_to
        with open(f"{safe_name}_{compile_to}.mlir", "w+") as f:
            f.write(module_str)
    else:
        flags = [
            "--iree-input-type=tm_tensor",
            "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            "--mlir-print-debuginfo",
            "--mlir-print-op-on-diagnostic=false",
            "--iree-llvmcpu-target-cpu-features=host",
            "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
            "--iree-llvmcpu-enable-microkernels",
            "--iree-llvmcpu-stack-allocation-limit=256000",
            "--iree-stream-resource-index-bits=64",
            "--iree-vm-target-index-bits=64",
            "--iree-vm-bytecode-module-strip-source-map=true",
            "--iree-util-zero-fill-elided-attrs",
            "--iree-vm-target-truncate-unsupported-floats",
            "--iree-codegen-check-ir-before-llvm-conversion=false",
            "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
            "--iree-opt-const-expr-hoisting=False",
        ]
        import iree.compiler as ireec

        flatbuffer_blob = ireec.compile_str(
            module_str,
            target_backends=["llvm-cpu"],
            extra_args=flags,
        )
        with open(f"{safe_name}.vmfb", "wb+") as f:
            f.write(flatbuffer_blob)


def run_vmfb_comparison(args):
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.mmap(
        config.vm_instance, "/home/dan/SHARK-Turbine/Llama_2_7b_chat_hf.vmfb"
    )
    ctx.add_vm_module(vm_module)
    ModuleCompiled = getattr(ctx.modules, vm_module.name)
    print(ModuleCompiled)

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_name,
        use_fast=False,
        use_auth_token=args.hf_auth_token,
    )
    initial_input = tokenizer(prompt, return_tensors="pt")
    example_input_id = initial_input.input_ids
    device_inputs = [ireert.asdevicearray(config.device, example_input_id)]

    results = ModuleCompiled["run_initialize"](*device_inputs)

    def format_out(results):
        return torch.tensor(results.to_host()[0][0])

    model = InferenceModel(args)

    def get_token_from_logits(logits):
        return torch.argmax(logits[:, -1, :], dim=1)

    base_model_results = model.base_model.forward(example_input_id)
    base_model_token = get_token_from_logits(base_model_results.logits)
    bm_pkv = base_model_results.past_key_values
    turbine_results = []
    torch_results = []
    turbine_results.append(format_out(results))
    torch_results.append(int(base_model_token))
    while base_model_token != 2:
        results = ModuleCompiled["run_forward"](results)
        step = ModuleCompiled["get_seq_step"]()
        pkv = ModuleCompiled["get_global_state"]().to_host()
        # print(f"turbine: {tokenizer.decode(format_out(results))}")
        base_model_results = model.base_model.forward(
            torch.unsqueeze(base_model_token, 0), past_key_values=bm_pkv
        )
        base_model_token = int(
            get_token_from_logits(base_model_results.logits)[0]
        )
        bm_pkv = base_model_results.past_key_values
        # print(f"pytorch: {tokenizer.decode(base_model_token)}")
        turbine_results.append(format_out(results))
        torch_results.append(base_model_token)

    print("\n\n")
    print("what is the best hardware company?")
    print("\n\n")

    print("turbine output: ")
    print(tokenizer.decode(turbine_results))
    print("torch output: ")
    print(tokenizer.decode(torch_results))


if __name__ == "__main__":
    args = parser.parse_args()
    if args.run_vmfb:
        run_vmfb_comparison(args)
    elif args.test:
        stateless_loop()
        test_class()
        test_stateless_against_torch()
    else:
        export_transformer_model(
            args.schema_path,
            args.hf_model_name,
            args.hf_auth_token,
            args.compile_to,
        )
