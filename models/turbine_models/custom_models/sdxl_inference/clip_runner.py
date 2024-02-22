import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer
from iree import runtime as ireert
import torch

parser = argparse.ArgumentParser()

# TODO move common runner flags to generic flag file
parser.add_argument(
    "--vmfb_path_1",
    type=str,
    default="",
    help="path to vmfb containing compiled module",
)
parser.add_argument(
    "--external_weight_path_1",
    type=str,
    default="",
    help="path to external weight parameters if model compiled without them",
)
parser.add_argument(
    "--vmfb_path_2",
    type=str,
    default="",
    help="path to vmfb containing compiled module",
)
parser.add_argument(
    "--external_weight_path_2",
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
    default="stabilityai/sdxl-turbo",
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
    default="a photograph of an astronaut riding a horse",
    help="prompt for clip model",
)
parser.add_argument(
    "--precision",
    type=str,
    default="fp32",
    help="fp16, fp32",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=77,
)


def run_clip(
    device,
    prompt,
    vmfb_path,
    hf_model_name,
    hf_auth_token,
    external_weight_path,
    max_length,
    index,
):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)

    tokenizer_1 = CLIPTokenizer.from_pretrained(
        hf_model_name,
        subfolder="tokenizer",
        token=hf_auth_token,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        hf_model_name,
        subfolder="tokenizer_2",
        token=hf_auth_token,
    )
    if index == 1:
        text_input = tokenizer_1(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        example_input = text_input.input_ids
        inp = [ireert.asdevicearray(runner.config.device, example_input)]
        results = runner.ctx.modules.compiled_clip["main"](*inp)
    elif index == 2:
        text_input = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        example_input = text_input.input_ids
        inp = [ireert.asdevicearray(runner.config.device, example_input)]
        results = runner.ctx.modules.compiled_clip["main"](*inp)
    else:
        print("Incorrect CLIP model index, please use 1 or 2")
        exit(1)

    return results


def run_torch_clip(hf_model_name, hf_auth_token, prompt, max_length=77):
    # TODO: Integrate with HFTransformerBuilder
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

    model_1 = CLIPTextModel.from_pretrained(
        hf_model_name,
        subfolder="text_encoder",
        token=hf_auth_token,
    )
    model_2 = CLIPTextModelWithProjection.from_pretrained(
        hf_model_name,
        subfolder="text_encoder_2",
        token=hf_auth_token,
    )
    tokenizer_1 = CLIPTokenizer.from_pretrained(
        hf_model_name,
        subfolder="tokenizer",
        token=hf_auth_token,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        hf_model_name,
        subfolder="tokenizer_2",
        token=hf_auth_token,
    )
    text_input_1 = tokenizer_1(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_2 = tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    example_input_1 = text_input_1.input_ids
    example_input_2 = text_input_2.input_ids

    results_1 = model_1.forward(example_input_1)[0]
    results_2 = model_2.forward(example_input_2)[0]
    np_torch_output_1 = results_1.detach().cpu().numpy()
    np_torch_output_2 = results_2.detach().cpu().numpy()
    return np_torch_output_1, np_torch_output_2


if __name__ == "__main__":
    args = parser.parse_args()
    turbine_output1 = run_clip(
        args.device,
        args.prompt,
        args.vmfb_path_1,
        args.hf_model_name,
        args.hf_auth_token,
        args.external_weight_path_1,
        args.max_length,
        index=1,
    )
    print(
        "TURBINE OUTPUT 1:",
        turbine_output1[0].to_host(),
        turbine_output1[0].to_host().shape,
        turbine_output1[0].to_host().dtype,
    )

    turbine_output2 = run_clip(
        args.device,
        args.prompt,
        args.vmfb_path_2,
        args.hf_model_name,
        args.hf_auth_token,
        args.external_weight_path_2,
        args.max_length,
        index=2,
    )
    print(
        "TURBINE OUTPUT 2:",
        turbine_output2[0].to_host(),
        turbine_output2[0].to_host().shape,
        turbine_output2[0].to_host().dtype,
    )
    if args.compare_vs_torch:
        print("generating torch output: ")
        from turbine_models.custom_models.sd_inference import utils

        torch_output1, torch_output2 = run_torch_clip(
            args.hf_model_name,
            args.hf_auth_token,
            args.prompt,
            args.precision,
            args.max_length,
        )
        print(
            "TORCH OUTPUT 1:", torch_output1, torch_output1.shape, torch_output1.dtype
        )
        err1 = utils.largest_error(torch_output1, turbine_output1[0])

        print(
            "TORCH OUTPUT 2:", torch_output2, torch_output2.shape, torch_output2.dtype
        )
        err2 = utils.largest_error(torch_output2, turbine_output2[0])
        print("Largest Error for CLIP 1: ", err1)
        print("Largest Error for CLIP 2: ", err2)
        assert err1 < 9e-5 and err2 < 9e-5
    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output1, turbine_output2 = (None, None)
