from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer
from iree import runtime as ireert
import torch
import numpy as np


def run_torch_clip(hf_model_name, hf_auth_token, prompt, max_length=64):
    # TODO: Integrate with HFTransformerBuilder
    from turbine_models.custom_models.sdxl_inference.clip import ClipModel

    model_1 = ClipModel(hf_model_name, hf_auth_token, index=1)
    model_2 = ClipModel(hf_model_name, hf_auth_token, index=2)
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

    results_1 = model_1.forward(example_input_1)
    results_2 = model_2.forward(example_input_2)
    np_torch_output_1 = results_1[0].detach().cpu().numpy().astype(np.float16)
    np_torch_output_2 = results_2[0].detach().cpu().numpy().astype(np.float16)
    return np_torch_output_1, np_torch_output_2


def run_prompt_encoder(
    args,
    input_ids,
    uncond_input_ids,
):
    prompt_encoder_runner = vmfbRunner(
        args.device, args.vmfb_path, args.external_weight_path
    )
    np.save("input0.npy", input_ids[0].numpy())
    np.save("input1.npy", input_ids[1].numpy())
    np.save("input2.npy", uncond_input_ids[0].numpy())
    np.save("input3.npy", uncond_input_ids[1].numpy())
    prompt_encoder_inputs = [
        ireert.asdevicearray(prompt_encoder_runner.config.device, input_ids[0]),
        ireert.asdevicearray(prompt_encoder_runner.config.device, input_ids[1]),
        ireert.asdevicearray(prompt_encoder_runner.config.device, uncond_input_ids[0]),
        ireert.asdevicearray(prompt_encoder_runner.config.device, uncond_input_ids[1]),
    ]
    encoded_outputs = prompt_encoder_runner.ctx.modules.compiled_clip["main"](
        *prompt_encoder_inputs
    )
    del prompt_encoder_inputs
    return encoded_outputs


if __name__ == "__main__":
    from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

    tokenizer_1 = CLIPTokenizer.from_pretrained(
        args.hf_model_name,
        subfolder="tokenizer",
        token=args.hf_auth_token,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.hf_model_name,
        subfolder="tokenizer_2",
        token=args.hf_auth_token,
    )
    text_input_ids_list = []
    uncond_input_ids_list = []

    # Tokenize prompt and negative prompt.
    tokenizers = [tokenizer_1, tokenizer_2]
    for tokenizer in tokenizers:
        text_inputs = tokenizer(
            args.prompt,
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input = tokenizer(
            args.negative_prompt,
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        uncond_input_ids = uncond_input.input_ids

        text_input_ids_list.extend([text_input_ids])
        uncond_input_ids_list.extend([uncond_input_ids])

    turbine_output1, turbine_output2 = run_prompt_encoder(
        args,
        text_input_ids_list,
        uncond_input_ids_list,
    )
    print(
        "TURBINE OUTPUT 1:",
        turbine_output1,
        turbine_output1.shape,
        turbine_output1.dtype,
    )

    print(
        "TURBINE OUTPUT 2:",
        turbine_output2,
        turbine_output2.shape,
        turbine_output2.dtype,
    )

    if args.compare_vs_torch:
        print("generating torch output: ")
        from turbine_models.custom_models.sd_inference import utils
        from turbine_models.custom_models.sdxl_inference.sdxl_prompt_encoder import (
            PromptEncoderModule,
        )

        torch_encoder_model = PromptEncoderModule(
            args.hf_model_name, args.precision, args.hf_auth_token
        )
        torch_output1, torch_output2 = torch_encoder_model.forward(
            *text_input_ids_list, *uncond_input_ids_list
        )
        np.save("torch_output1.npy", torch_output1)
        np.save("torch_output2.npy", torch_output2)
        print(
            "TORCH OUTPUT 1:", torch_output1, torch_output1.shape, torch_output1.dtype
        )

        print(
            "TORCH OUTPUT 2:", torch_output2, torch_output2.shape, torch_output2.dtype
        )
        rtol = 4e-2
        atol = 4e-2

        np.testing.assert_allclose(
            torch_output1, turbine_output1.to_host(), rtol, atol, verbose=True
        )
        np.testing.assert_allclose(
            torch_output2, turbine_output2.to_host(), rtol, atol, verbose=True
        )
        print("Passed!")
    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output1, turbine_output2 = (None, None)
