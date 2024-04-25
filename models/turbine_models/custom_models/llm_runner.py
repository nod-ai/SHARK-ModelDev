import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import AutoTokenizer, AutoModelForCausalLM
from iree import runtime as ireert
import torch
import time
from turbine_models.custom_models.llm_optimizations.streaming_llm.modify_llama import (
    enable_llama_pos_shift_attention,
)

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
    "--streaming_llm",
    action="store_true",
    help="Use streaming LLM mode for longer context and low memory usage.",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="hi what are you?",
    help="prompt for llm model",
)
parser.add_argument(
    "--chat_mode",
    action="store_true",
    help="Runs an interactive CLI chat mode.",
)
parser.add_argument(
    "--chat_sys_prompt",
    type=str,
    default="""<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n <</SYS>>\n\n
""",
    help="System prompt used for interactive chat mode.",
)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<s>", "</s>"
DEFAULT_CHAT_SYS_PROMPT = """<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n <</SYS>>\n\n
"""


def append_user_prompt(history, input_prompt):
    user_prompt = f"{B_INST} {input_prompt} {E_INST}"
    history += user_prompt
    return history


def append_bot_prompt(history, input_prompt):
    user_prompt = f"{B_SYS} {input_prompt}{E_SYS} {E_SYS}"
    history += user_prompt
    return history


class SharkLLM(object):
    def __init__(self, device, vmfb_path, external_weight_path, streaming_llm=False):
        self.runner = vmfbRunner(
            device=device,
            vmfb_path=vmfb_path,
            external_weight_path=external_weight_path,
        )
        if streaming_llm:
            self.model = self.runner.ctx.modules.streaming_state_update
        else:
            self.model = self.runner.ctx.modules.state_update
        self.first_input = True
        self.num_tokens = 0
        self.last_prompt = None
        self.streaming_llm = streaming_llm
        self.prev_token_len = 0

    def format_out(self, results):
        return torch.tensor(results.to_host()[0][0])

    def evict_kvcache_space(self):
        self.model["evict_kvcache_space"]()

    def generate(self, input_ids):
        # TODO: Replace with args.
        if self.streaming_llm and self.model["get_seq_step"]() > 600:
            print("Evicting cache space!")
            self.model["evict_kvcache_space"]()
        turbine_results = []
        # Only need not seen token for init cache
        # Because we have stored the res in KV-cache.
        token_len = input_ids.shape[-1]
        if self.streaming_llm:
            token_slice = max(self.prev_token_len - 1, 0)
            input_ids = input_ids[:, token_slice:]
        inputs = [ireert.asdevicearray(self.runner.config.device, input_ids)]
        if self.first_input or not self.streaming_llm:
            s = time.time()
            results = self.model["run_initialize"](*inputs)  # example_input_id
            e = time.time()
            print(
                f"num_tokens: {token_len}, time_taken={e-s}, tok/second:{token_len/(e-s)}"
            )
            token_len += 1
            self.first_input = False
        else:
            s = time.time()
            results = self.model["run_cached_initialize"](*inputs)  # example_input_id
            e = time.time()
            print(
                f"Cached num_tokens: {token_len}, time_taken={e-s}, tok/second:{token_len/(e-s)}"
            )
            token_len += 1
        s = time.time()
        turbine_results.append(self.format_out(results))
        while self.format_out(results) != 2:
            if self.streaming_llm and self.model["get_seq_step"]() > 600:
                print("Evicting cache space!")
                self.model["evict_kvcache_space"]()
            results = self.model["run_forward"](results)
            # uncomment to see tokens as they are emitted
            # print(f"turbine: {tokenizer.decode(self.format_out(results))}")
            turbine_results.append(self.format_out(results))
        e = time.time()
        decoded_tokens = len(turbine_results)
        print(
            f"Decode num_tokens: {decoded_tokens}, time_taken={e-s}, tok/second:{decoded_tokens/(e-s)}"
        )
        self.prev_token_len = token_len + decoded_tokens
        return turbine_results


def run_llm(
    device,
    prompt,
    vmfb_path,
    hf_model_name,
    hf_auth_token,
    external_weight_path,
    streaming_llm=False,
    chat_mode=False,
    chat_sys_prompt=DEFAULT_CHAT_SYS_PROMPT,
    tokenizer=None,
):
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            use_fast=False,
            token=hf_auth_token,
        )
    llm = SharkLLM(
        device=device,
        vmfb_path=vmfb_path,
        external_weight_path=external_weight_path,
        streaming_llm=streaming_llm,
    )
    if not chat_mode:
        prompt = append_user_prompt(chat_sys_prompt, prompt)
        initial_input = tokenizer(prompt, return_tensors="pt")
        example_input_id = initial_input.input_ids
        turbine_results = llm.generate(example_input_id)
        return tokenizer.decode(turbine_results)
    prompt = chat_sys_prompt
    while True:
        user_prompt = input("User prompt: ")
        prompt = append_user_prompt(prompt, user_prompt)
        initial_input = tokenizer(prompt, return_tensors="pt")
        example_input_id = initial_input.input_ids
        result = llm.generate(example_input_id)
        bot_response = tokenizer.decode(result, skip_special_tokens=True)
        print(f"\nBOT: {bot_response}\n")
        prompt = append_bot_prompt(prompt, bot_response)


def run_torch_llm(
    hf_model_name,
    hf_auth_token,
    prompt,
    streaming_llm=False,
    chat_sys_prompt=DEFAULT_CHAT_SYS_PROMPT,
    model=None,
    tokenizer=None,
):
    if model == None:
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=torch.float,
            token=hf_auth_token,
        )
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name,
            use_fast=False,
            token=hf_auth_token,
        )
    if streaming_llm is True:
        enable_llama_pos_shift_attention(model)

    def get_token_from_logits(logits):
        return torch.argmax(logits[:, -1, :], dim=1)

    prompt = append_user_prompt(chat_sys_prompt, prompt)
    initial_input = tokenizer(prompt, return_tensors="pt")
    example_input_id = initial_input.input_ids

    model_results = model.forward(example_input_id)
    model_token = get_token_from_logits(model_results.logits)

    pkv = model_results.past_key_values

    torch_results = []
    torch_results.append(int(model_token))
    while model_token != 2:
        model_results = model.forward(
            torch.unsqueeze(model_token, 0), past_key_values=pkv
        )
        model_token = get_token_from_logits(model_results.logits)
        pkv = model_results.past_key_values
        torch_results.append(int(model_token[0]))

    return tokenizer.decode(torch_results)


if __name__ == "__main__":
    args = parser.parse_args()
    print("generating turbine output: ")
    turbine_output = run_llm(
        args.device,
        args.prompt,
        args.vmfb_path,
        args.hf_model_name,
        args.hf_auth_token,
        args.external_weight_path,
        args.streaming_llm,
        args.chat_mode,
        args.chat_sys_prompt,
    )
    print(turbine_output)
    if args.compare_vs_torch:
        print("generating torch output: ")
        torch_output = run_torch_llm(
            args.hf_model_name,
            args.hf_auth_token,
            args.prompt,
            args.streaming_llm,
            args.chat_sys_prompt,
        )
        print(torch_output)
