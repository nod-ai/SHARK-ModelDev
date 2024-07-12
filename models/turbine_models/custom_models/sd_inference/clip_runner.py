import argparse
from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer
from iree import runtime as ireert
import torch


def run_clip(
    device, prompt, vmfb_path, hf_model_name, hf_auth_token, external_weight_path
):
    runner = vmfbRunner(device, vmfb_path, external_weight_path)

    if "google/t5" in hf_model_name:
        from transformers import T5Tokenizer, T5Model

        tokenizer = T5Tokenizer.from_pretrained(hf_model_name)
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
    # TODO: Integrate with HFTransformerBuilder
    else:
        if "openai" in hf_model_name:
            from transformers import CLIPProcessor
            import requests

            tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            text_input = tokenizer(
                text=prompt,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            hf_subfolder = "tokenizer"

            tokenizer = CLIPTokenizer.from_pretrained(
                hf_model_name,
                subfolder=hf_subfolder,
                token=hf_auth_token,
            )

            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
    example_input = text_input.input_ids
    inp = [ireert.asdevicearray(runner.config.device, example_input)]

    if "google/t5" in hf_model_name:
        inp += [ireert.asdevicearray(runner.config.device, example_input)]
    results = runner.ctx.modules.compiled_text_encoder["encode_tokens"](*inp)
    return results


def run_torch_clip(hf_model_name, hf_auth_token, prompt):
    if "google/t5" in hf_model_name:
        from transformers import T5Tokenizer, T5Model

        tokenizer = T5Tokenizer.from_pretrained(hf_model_name)
        model = T5Model.from_pretrained(hf_model_name)
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
    # TODO: Integrate with HFTransformerBuilder
    else:
        if hf_model_name == "openai/clip-vit-large-patch14":
            from transformers import CLIPProcessor

            tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            hf_subfolder = ""  # CLIPProcessor does not have a subfolder
            from transformers import CLIPTextModel

            model = CLIPTextModel.from_pretrained(
                hf_model_name,
                subfolder=hf_subfolder,
                token=hf_auth_token,
            )
            text_input = tokenizer(
                text=prompt,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
        else:
            hf_subfolder = "text_encoder"

            tokenizer = CLIPTokenizer.from_pretrained(
                hf_model_name,
                subfolder="tokenizer",
                token=hf_auth_token,
            )

            from transformers import CLIPTextModel

            model = CLIPTextModel.from_pretrained(
                hf_model_name,
                subfolder=hf_subfolder,
                token=hf_auth_token,
            )
            text_input = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
    example_input = text_input.input_ids

    if "google/t5" in hf_model_name:
        results = model.forward(example_input, decoder_input_ids=example_input)[0]
    else:
        results = model.forward(example_input)[0]
    np_torch_output = results.detach().cpu().numpy()
    return np_torch_output


if __name__ == "__main__":
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

    turbine_output = run_clip(
        args.device,
        args.prompt,
        args.vmfb_path,
        args.hf_model_name,
        args.hf_auth_token,
        args.external_weight_path,
    )
    print(
        "TURBINE OUTPUT:",
        turbine_output[0].to_host(),
        turbine_output[0].to_host().shape,
        turbine_output[0].to_host().dtype,
    )
    if args.compare_vs_torch:
        print("generating torch output: ")
        from turbine_models.custom_models.sd_inference import utils

        torch_output = run_torch_clip(
            args.hf_model_name, args.hf_auth_token, args.prompt
        )
        print("TORCH OUTPUT:", torch_output, torch_output.shape, torch_output.dtype)
        err = utils.largest_error(torch_output, turbine_output[0])
        print("Largest Error: ", err)
        assert err < 9e-5
    # TODO: Figure out why we occasionally segfault without unlinking output variables
    turbine_output = None
