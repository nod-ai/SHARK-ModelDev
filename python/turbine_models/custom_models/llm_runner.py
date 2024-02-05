import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import AutoTokenizer
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
    default="""<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
""",
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
        self.min_token = 0
        self.max_token = 1024
        self.last_prefill_time = -1.0
        self.last_prompt_decode_time = -1.0
        self.last_num_tokens_decoded = -1
        self.last_num_tokens_prefill = -1

    def set_min_token(self, min_token):
        self.min_token = min_token

    def set_max_token(self, max_token):
        self.max_token = max_token

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
            prefill_start_time = time.time()
            results = self.model["run_initialize"](*inputs)  # example_input_id
            prefill_end_time = time.time()
            self.last_num_tokens_prefill = token_len
            self.last_prefill_time = prefill_end_time - prefill_start_time
            token_len += 1
            self.first_input = False
        else:
            prefill_start_time = time.time()
            results = self.model["run_cached_initialize"](*inputs)  # example_input_id
            prefill_end_time = time.time()
            self.last_num_tokens_prefill = token_len
            self.last_prefill_time = prefill_end_time - prefill_start_time
            token_len += 1
        decode_start_time = time.time()
        turbine_results.append(self.format_out(results))
        for _ in range(self.max_token):
            if self.streaming_llm and self.model["get_seq_step"]() > 600:
                print("Evicting cache space!")
                self.model["evict_kvcache_space"]()
            results = self.model["run_forward"](results)
            # uncomment to see tokens as they are emitted
            # print(f"turbine: {tokenizer.decode(self.format_out(results))}")
            turbine_results.append(self.format_out(results))
            if self.format_out(results) == 2 and len(turbine_results) >= self.min_token:
                break
        decode_end_time = time.time()
        decoded_tokens = len(turbine_results)
        self.last_prompt_decode_time = decode_end_time - decode_start_time
        self.last_num_tokens_decoded = decoded_tokens
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
):
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
        print(
            f"Prefill num_tokens : {llm.last_num_tokens_prefill}, time_taken: {llm.last_prefill_time}, tok/second: {llm.last_num_tokens_prefill/llm.last_prefill_time}"
        )
        print(
            f"Decode num_tokens : {llm.last_num_tokens_decoded}, time_taken: {llm.last_prompt_decode_time}, tok/second: {llm.last_num_tokens_decoded/llm.last_prompt_decode_time}"
        )
        prompt = append_bot_prompt(prompt, bot_response)


def run_torch_llm(hf_model_name, hf_auth_token, prompt, streaming_llm=False):
    from turbine_models.model_builder import HFTransformerBuilder
    from transformers import AutoModelForCausalLM

    model_builder = HFTransformerBuilder(
        example_input=None,
        hf_id=hf_model_name,
        auto_model=AutoModelForCausalLM,
        hf_auth_token=hf_auth_token,
        auto_tokenizer=AutoTokenizer,
    )
    model_builder.build_model()
    if streaming_llm is True:
        enable_llama_pos_shift_attention(model_builder.model)

    def get_token_from_logits(logits):
        return torch.argmax(logits[:, -1, :], dim=1)

    initial_input = model_builder.tokenizer(prompt, return_tensors="pt")
    example_input_id = initial_input.input_ids

    model_results = model_builder.model.forward(example_input_id)
    model_token = get_token_from_logits(model_results.logits)

    pkv = model_results.past_key_values

    torch_results = []
    torch_results.append(int(model_token))
    while model_token != 2:
        model_results = model_builder.model.forward(
            torch.unsqueeze(model_token, 0), past_key_values=pkv
        )
        model_token = get_token_from_logits(model_results.logits)
        pkv = model_results.past_key_values
        torch_results.append(int(model_token[0]))

    return model_builder.tokenizer.decode(torch_results)


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
            args.hf_model_name, args.hf_auth_token, args.prompt
        )
        print(torch_output)
