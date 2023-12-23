import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import AutoTokenizer
from iree import runtime as ireert
import torch
import time
parser = argparse.ArgumentParser()

# TODO move common runner flags to generic flag file
parser.add_argument(
    "--vmfb_path", type=str, default="", help="path to vmfb containing compiled module"
)
parser.add_argument(
    "--external_weight_path",
    type=str,
    default="",
    help="path to external weight parameters if model compiled without them",
)
parser.add_argument(
    "--compare_vs_torch",
    action="store_true",
    help="Runs both turbine vmfb and a torch model to compare results",
)
parser.add_argument(
    "--hf_model_name",
    type=str,
    help="HF model name",
    default="meta-llama/Llama-2-7b-chat-hf",
)
parser.add_argument(
    "--hf_auth_token",
    type=str,
    help="The Hugging face auth token, required for some models",
)
parser.add_argument(
    "--device",
    type=str,
    default="local-task",
    help="local-sync, local-task, cuda, vulkan, rocm",
)
parser.add_argument(
    "--init_cache",
    type=bool,
    default=False,
    help="Use KV-Cache in between user prompts/multi-dialogue.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="""<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n <</SYS>>\n\n
""",
    help="prompt for llm model",
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
def append_user_prompt(history, input_prompt):
    user_prompt = f"{B_INST} {input_prompt} {E_INST}"
    history += user_prompt
    return history

def append_bot_prompt(history, input_prompt):
    user_prompt = f"{B_SYS} {input_prompt} {E_SYS}"
    history += user_prompt
    return history

class SharkLLM(object):
    def __init__(self, device, vmfb_path, external_weight_path, init_cache=False):
        self.runner = vmfbRunner(
            device=device, vmfb_path=vmfb_path, external_weight_path=external_weight_path
        )
        self.first_input = True
        self.num_tokens = 0
        self.last_prompt = None
        self.init_cache = init_cache
        self.prev_token_len = 0

    def format_out(self, results):
        return torch.tensor(results.to_host()[0][0])

    def generate(self, input_ids):
        turbine_results = []
        # Only need not seen token for init cache
        # Because we have stored the res in KV-cache.
        token_len = input_ids.shape[-1]
        if self.init_cache:
            input_ids = input_ids[:, self.prev_token_len:]
        inputs = [ireert.asdevicearray(self.runner.config.device, input_ids)]
        if (self.first_input or not self.init_cache):
            s = time.time()
            results = self.runner.ctx.modules.state_update["run_initialize"](
                *inputs
            )  # example_input_id
            e = time.time()
            print(f"num_tokens: {token_len}, time_taken={e-s}, tok/second:{token_len/(e-s)}")
            token_len += 1
            self.first_input = False
        else:
            s = time.time()
            results = self.runner.ctx.modules.state_update["run_cached_initialize"](
                *inputs
            )  # example_input_id
            e = time.time()
            token_len += 1
            print(f"Cached num_tokens: {token_len}, time_taken={e-s}, tok/second:{token_len/(e-s)}")
        s = time.time()
        predecode_tokens = token_len
        while self.format_out(results) != 2:
            results = self.runner.ctx.modules.state_update["run_forward"](results)
            # uncomment to see tokens as they are emitted
            # print(f"turbine: {tokenizer.decode(self.format_out(results))}")
            turbine_results.append(self.format_out(results))
            token_len += 1
        e = time.time()
        decoded_tokens = token_len - predecode_tokens
        print(f"Decode num_tokens: {decoded_tokens}, time_taken={e-s}, tok/second:{decoded_tokens/(e-s)}")
        self.prev_token_len = token_len
        return turbine_results

def run_llm(
    device, system_prompt, vmfb_path, hf_model_name, hf_auth_token, external_weight_path, init_cache
):
    runner = vmfbRunner(
        device=device, vmfb_path=vmfb_path, external_weight_path=external_weight_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
        token=hf_auth_token,
    )
    llm = SharkLLM(device=device, vmfb_path=vmfb_path, external_weight_path=external_weight_path, init_cache=init_cache)
    prompt = system_prompt
    while True:
        user_prompt = input("User prompt: ")
        prompt = append_user_prompt(prompt, user_prompt)
        initial_input = tokenizer(prompt, return_tensors="pt")
        example_input_id = initial_input.input_ids
        result = llm.generate(example_input_id)
        bot_response = tokenizer.decode(result)
        print(f"\nBOT: {bot_response}\n")
        prompt = append_bot_prompt(prompt, bot_response)

if __name__ == "__main__":
    args = parser.parse_args()
    print("generating turbine output: ")
    run_llm(
        args.device,
        args.prompt,
        args.vmfb_path,
        args.hf_model_name,
        args.hf_auth_token,
        args.external_weight_path,
        args.init_cache,
    )
