import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import AutoTokenizer
from iree import runtime as ireert
import torch
import time
from turbine_models.custom_models.llm_optimizations.streaming_llm.modify_llama import (
    enable_llama_pos_shift_attention,
)
from turbine_models.custom_models.llm_runner import parser, SharkLLM
import os
import json

parser.add_argument(
    "--benchmark_dataset_path",
    type=str,
    default=f"{os.path.dirname(os.path.realpath(__file__))}/benchmark_prompts.json",
    help="path to benchmarking dataset",
)
parser.add_argument(
    "--benchmark_output_path",
    type=str,
    default=f"{os.getcwd()}/benchmark_e2e_results.json",
    help="path to benchmarking dataset",
)


B_INST, E_INST = "[INST]", "[/INST]"


def append_user_prompt(history, input_prompt):
    if len(input_prompt) == 0:
        return history
    user_prompt = f"{B_INST} {input_prompt} {E_INST}"
    history += user_prompt
    return history


def load_dataset(dataset_path):
    dataset = []
    with open(dataset_path) as f:
        dataset = json.load(f)
    if len(dataset) <= 0:
        raise ValueError("Dataset is empty, or did not read dataset correctly.")
    return dataset


def run_llm_benchmark(
    device,
    vmfb_path,
    hf_model_name,
    hf_auth_token,
    external_weight_path,
    dataset_path,
    output_path,
    streaming_llm=False,
):
    # TODO: Support streamingLLM benchmarking, need streamingLLM to be able to reset history/seq_len to 0.
    if streaming_llm:
        raise ValueError("Streaming LLM currently not supported for benchmarking.")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
        token=hf_auth_token,
    )
    dataset = load_dataset(dataset_path)
    result_dicts = []
    llm = SharkLLM(
        device=device,
        vmfb_path=vmfb_path,
        external_weight_path=external_weight_path,
        streaming_llm=streaming_llm,
    )
    for data in dataset:
        llm.set_min_token(data["num_tokens_to_generate"])
        llm.set_max_token(data["num_tokens_to_generate"])
        running_token_decode_count = 0
        running_token_decode_time = 0.0
        running_token_prefill_count = 0
        running_token_prefill_time = 0.0
        for _ in range(data["num_iterations"]):
            prompt = data["system_prompt"]
            prompt = append_user_prompt(prompt, data["user_prompt"])
            initial_input = tokenizer(prompt, return_tensors="pt")
            example_input_id = initial_input.input_ids
            result = llm.generate(example_input_id)
            bot_response = tokenizer.decode(result, skip_special_tokens=True)
            running_token_decode_count += llm.last_num_tokens_decoded
            running_token_decode_time += llm.last_prompt_decode_time
            running_token_prefill_count += llm.last_num_tokens_prefill
            running_token_prefill_time += llm.last_prefill_time
        prefill_tokens = running_token_prefill_count / data["num_iterations"]
        prefill_speed = running_token_prefill_count / running_token_prefill_time
        decoded_tokens = running_token_decode_count / data["num_iterations"] - 1
        decode_speed = running_token_decode_count / running_token_decode_time
        result_dicts.append(
            {
                "prompt_id": data["id"],
                "system_prompt": data["system_prompt"],
                "user_prompt": data["user_prompt"],
                "prefill_tokens": prefill_tokens,
                "prefill_speed(tok/s)": prefill_speed,
                "decoded_tokens": decoded_tokens,
                "decode_speed(tok/s)": decode_speed,
                "num_iterations": data["num_iterations"],
                "response": bot_response,
            }
        )
    with open(output_path, "w") as f:
        json_results = json.dumps(result_dicts, indent=2)
        f.write(json_results)
    return output_path


if __name__ == "__main__":
    args = parser.parse_args()
    print("generating turbine output: ")
    turbine_output_file = run_llm_benchmark(
        args.device,
        args.vmfb_path,
        args.hf_model_name,
        args.hf_auth_token,
        args.external_weight_path,
        args.benchmark_dataset_path,
        args.benchmark_output_path,
        args.streaming_llm,
    )
