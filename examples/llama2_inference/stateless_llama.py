import os
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
AUTH_TOKEN = "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"
from torch.fx import (
    GraphModule,
)
import collections
from torch._export.constraints import constrain_as_size, constrain_as_value
from shark_turbine.aot import *
from iree.compiler.ir import Context
BATCH_SIZE = 1
MAX_STEP_SEQ = 4095
MAGIC_SIZE = 5 
def import_compiler(gm: GraphModule, example_inputs):
    imp = FxImporter()
    gm = turbine_cpu_pass_pipeline(gm, example_inputs)

    try:
        imp.import_graph_module(gm)
    finally:
        print(imp.module)
    imp.module.operation.verify()
    return gm

class InferenceModel(torch.nn.Module):
    def __init__(self, base_model_name="meta-llama/Llama-2-7b-chat-hf", 
                 state_schema_path="test/dynamo/llama2_state_schema.json"):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float,
            use_auth_token=AUTH_TOKEN,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_fast=False,
            use_auth_token=AUTH_TOKEN,
        )
        self.base_model_name = base_model_name
        if os.path.exists(state_schema_path):
            with open(state_schema_path, "r+") as f:
                self.state_schema = pytree.treespec_loads(f.read())
        else:
            self.generate_state_schema()

    def get_sample_input(self):
        prompt = ("""<s>[INST] <<SYS>>
        Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
        """)
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
                BATCH_SIZE, sample_shape[1], MAX_STEP_SEQ, sample_shape[3], 
                dtype=x.dtype), 
            initial_results.past_key_values) 
        _, self.state_schema = pytree.tree_flatten(pkv)

    def write_schema_to_file(self, schema_path=None):
        if schema_path == None:
            schema_path = f"{self.model_name.split('/')[-1]}_schema.json",
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
            constraints= [
                dynamic_dim(example_state[0], 2) < MAX_STEP_SEQ
            ] + [
                (dynamic_dim(x, 2) == (dynamic_dim(example_state[0], 2))) for x in example_state[1:]
            ],
        )
        g_forward, guards_forward = exp_forward(example_token, *example_state)
        if not compile_dynamo:
            return g_initialize, guards_initialize, g_forward, guards_forward
        g_initialize = import_compiler(g_initialize, [example_input_id])
        g_forward = import_compiler(g_forward, [example_token, *example_state])

        return g_initialize, guards_initialize, g_forward, guards_forward


def stateless_loop():
    def unpack_tensor(pkv, seq_step):
        return [pkv[i,:,:,:seq_step+1,:] for i in range(64)]
    # example of loop using stateless. Roughly what we need to model with the iree emission
    model = InferenceModel()
    input_ids = model.get_sample_input()

    example_token0, *state1_flat = model.initialize(input_ids)
    seq_step = state1_flat[0].shape[2]
    shape_default = list(state1_flat[0].shape)
    shape_default[2] = MAX_STEP_SEQ
    shape_default = tuple([64]+shape_default)
    print(shape_default)
    session_state = torch.zeros(shape_default, dtype=state1_flat[0].dtype)
    for i in range(64):
        session_state[i,:,:,:seq_step,:] = state1_flat[i]

    token = example_token0
    for i in range(15):
        next_input_token = torch.reshape(token, [1,1])
        token, *state1_flat_update = model.forward(next_input_token, *unpack_tensor(session_state, seq_step)) 
        
        for i in range(64):
            session_state[i,:,:,seq_step:seq_step+1,:] = state1_flat_update[i] 
        seq_step+=1

def slice_up_to_step(global_pkv, seq_step, kv_count=MAGIC_SIZE):
    #again maybe we can do a loop inside torch and just pass the entire kv_count tensor from here?
    all_pkv_tensors = []
    for i in range(kv_count):
        all_pkv_tensors.append(
                IREE.tensor_slice(global_pkv,
                                  i,
                                  0,
                                  0,
                                  (0, seq_step),
                                  0) #sequence context dim
                                  )

    return all_pkv_tensors

def update_state(state, state_updates, seq_step, kv_count=MAGIC_SIZE):
    all_updates = []
    # maybe instead of a for loop here we can concatenate the pkv from inside pytorch
    for i in range(kv_count):
        update = IREE.tensor_reshape(state_updates[i],1,1,32,1,128)
        all_updates.append(
               IREE.tensor_update(
                   state, 
                   update,
                   i, 0, 0, seq_step, 0))
    return all_updates

import torch.nn as nn
class SimpleParams(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3,5)
    def initialize(self, input_token):
        out_0 = [input_token *2]
        out_1 = [torch.zeros(size=[1,32,64,128]) for x in range(MAGIC_SIZE)]
        return out_0, *out_1
    def forward(self, token, *state):
        token = token*1
        state = [x*2.0 for x in state]

        state1_flat = [torch.ones(size=(1,32,1,128)) for x in range(MAGIC_SIZE)]
        return token, *state1_flat

#mod = InferenceModel() 
mod = SimpleParams()
seq_step = AbstractIndex
global_pkv = torch.zeros(size=(MAGIC_SIZE, 1, 32, 4095, 128), dtype=torch.float32)
seq_incr = AbstractIndex
# (64, 1, 32, 4095, 128)
class StateUpdateModule(CompiledModule):
    params = export_parameters(mod)
    global_state = export_global(global_pkv, mutable=True, initialize=False)
    global_seq_step = export_global(seq_step, mutable=True, initialize=False)
    initialize = jittable(mod.initialize)
    forward = jittable(mod.forward)
    def run(self, x=AbstractTensor(1,1, dtype=torch.int32)):
        token, *state = self.initialize(x)
        updates=[]
        self.global_seq_step = IREE.tensor_dim(state[0], 3) #3rd dimension of arbitrarily 0th kv tensor
        for i in range(MAGIC_SIZE):
            slice_of_state = IREE.tensor_reshape(state[i],1,1,32,self.global_seq_step,128)
            updates.append(IREE.tensor_update(self.global_state, slice_of_state,i,0,0,0,0))

    def run_forward(self, x=AbstractTensor(1,3, dtype=torch.int32)):
        state_arg = slice_up_to_step(self.global_state, self.global_seq_step)
        token, *state_update = self.forward(x)
        res = update_state(self.global_state, state_update, self.global_seq_step) 
#        self.global_seq_step = self.global_seq_step+ self.seq_incr



def test_class():
    inst = StateUpdateModule(context=Context())
    module_str = str(CompiledModule.get_mlir_module(inst))
    with open("output.mlir", "w+") as f:
        f.write(module_str)
        
def test_against_golden():
    def unpack_tensor(pkv, seq_step):
        return [pkv[i,:,:,:seq_step+1,:] for i in range(64)]
    model = InferenceModel()
    def get_token_from_logits(logits):
        return torch.argmax(logits[:,-1,:], dim=1)
    input_ids = model.get_sample_input()
    example_token0, *state1_flat = model.initialize(input_ids)
    seq_step = state1_flat[0].shape[2]
    shape_default = list(state1_flat[0].shape)
    shape_default[2] = MAX_STEP_SEQ
    shape_default = tuple([64]+shape_default)
    session_state = torch.zeros(shape_default, dtype=state1_flat[0].dtype)
    for i in range(64):
        session_state[i,:,:,:seq_step,:] = state1_flat[i]
    #get base model results
    base_model_results = model.base_model.forward(input_ids)
    base_model_token = get_token_from_logits(base_model_results.logits) 
    assert(example_token0 == base_model_token)
    base_model_pkv = base_model_results.past_key_values
    token = example_token0
    for i in range(15):
        next_input_token = torch.reshape(token, [1,1])
        print(f"next_input_token {next_input_token}")
        base_results = model.base_model.forward(next_input_token, past_key_values=base_model_pkv) 
        base_token = get_token_from_logits(base_results.logits)
        base_model_pkv = base_results.past_key_values

        state_as_list = unpack_tensor(session_state, seq_step)
        token, *state1_flat_update = model.forward(next_input_token, *unpack_tensor(session_state, seq_step)) 
        for i in range(64):
            session_state[i,:,:,seq_step:seq_step+1,:] = state1_flat_update[i] 
        seq_step+=1

        print(f"stateless_token {model.tokenizer.decode(token)}")
        print(f"base_token {model.tokenizer.decode(base_token)}")
        print()
        assert(token==base_token)

def test_compile():
    model = InferenceModel()
    input_ids = model.get_sample_input()
    g_initialize, guards_initialize, g_forward, guards_forward = model.compile(input_ids, True)


if __name__ == "__main__":
#    stateless_loop()
#    test_compile()
    test_class() 
#    test_against_golden()
