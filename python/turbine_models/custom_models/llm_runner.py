import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import AutoTokenizer
from iree import runtime as ireert
import torch

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
    "--prompt",
    type=str,
    default="""<s>[INST] <<SYS>>
Be concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> hi what are you? [/INST]
""",
    help="prompt for llm model",
)


def run_llm(
    device, prompt, vmfb_path, hf_model_name, hf_auth_token, external_weight_path
):
    runner = vmfbRunner(
        device=device, vmfb_path=vmfb_path, external_weight_path=external_weight_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_name,
        use_fast=False,
        token=hf_auth_token,
    )
    initial_input = tokenizer(prompt, return_tensors="pt")
    example_input_id = initial_input.input_ids
    inputs = [ireert.asdevicearray(runner.config.device, example_input_id)]
    results = runner.ctx.modules.state_update["run_initialize"](
        *inputs
    )  # example_input_id)

    def format_out(results):
        return torch.tensor(results.to_host()[0][0])

    turbine_results = []
    turbine_results.append(format_out(results))
    while format_out(results) != 2:
        results = runner.ctx.modules.state_update["run_forward"](results)
        # uncomment to see tokens as they are emitted
        # print(f"turbine: {tokenizer.decode(format_out(results))}")
        turbine_results.append(format_out(results))

    return tokenizer.decode(turbine_results)


def run_torch_llm(hf_model_name, hf_auth_token, prompt):
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
    )
    print(turbine_output)
    if args.compare_vs_torch:
        print("generating torch output: ")
        torch_output = run_torch_llm(
            args.hf_model_name, args.hf_auth_token, args.prompt
        )
        print(torch_output)
