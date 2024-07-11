from turbine_models.model_runner import vmfbRunner
from turbine_models.custom_models.sd3_inference.text_encoder_impls import (
    SD3Tokenizer,
    T5XXLTokenizer,
    SDXLClipGTokenizer,
)
from iree import runtime as ireert
import torch
import numpy as np


def run_prompt_encoder(
    vmfb_path,
    device,
    external_weight_path,
    input_ids,
    uncond_input_ids,
):
    prompt_encoder_runner = vmfbRunner(device, vmfb_path, external_weight_path)
    # np.save("input0.npy", input_ids[0].numpy())
    # np.save("input1.npy", input_ids[1].numpy())
    # np.save("input2.npy", input_ids[2].numpy())
    # np.save("input3.npy", uncond_input_ids[0].numpy())
    # np.save("input4.npy", uncond_input_ids[1].numpy())
    # np.save("input5.npy", uncond_input_ids[2].numpy())
    prompt_encoder_inputs = [
        ireert.asdevicearray(prompt_encoder_runner.config.device, input_ids[0]),
        ireert.asdevicearray(prompt_encoder_runner.config.device, input_ids[1]),
        ireert.asdevicearray(prompt_encoder_runner.config.device, input_ids[2]),
        ireert.asdevicearray(prompt_encoder_runner.config.device, uncond_input_ids[0]),
        ireert.asdevicearray(prompt_encoder_runner.config.device, uncond_input_ids[1]),
        ireert.asdevicearray(prompt_encoder_runner.config.device, uncond_input_ids[2]),
    ]
    encoded_outputs = prompt_encoder_runner.ctx.modules.compiled_text_encoder[
        "encode_tokens"
    ](*prompt_encoder_inputs)
    for i in encoded_outputs:
        i = i.to_host()
    del prompt_encoder_inputs
    return encoded_outputs


def run_tokenize(
    tokenizer,
    prompt,
    negative_prompt,
):
    prompt_tokens_dict = tokenizer.tokenize_with_weights(prompt)
    neg_prompt_tokens_dict = tokenizer.tokenize_with_weights(negative_prompt)
    text_input_ids_list = list(prompt_tokens_dict.values())
    uncond_input_ids_list = list(neg_prompt_tokens_dict.values())
    return text_input_ids_list, uncond_input_ids_list


if __name__ == "__main__":
    from turbine_models.custom_models.sd3_inference.sd3_cmd_opts import args

    tokenizer = SD3Tokenizer()

    text_input_ids_list, uncond_input_ids_list = run_tokenize(
        tokenizer,
        args.prompt,
        args.negative_prompt,
    )
    turbine_output1, turbine_output2 = run_prompt_encoder(
        args.vmfb_path,
        args.rt_device,
        args.external_weight_path,
        text_input_ids_list,
        uncond_input_ids_list,
    )
    print(
        "TURBINE OUTPUT 1:",
        turbine_output1.to_host(),
        turbine_output1.shape,
        turbine_output1.dtype,
    )

    print(
        "TURBINE OUTPUT 2:",
        turbine_output2.to_host(),
        turbine_output2.shape,
        turbine_output2.dtype,
    )

    if args.compare_vs_torch:
        print("generating torch output: ")
        from turbine_models.custom_models.sd_inference import utils
        from turbine_models.custom_models.sd3_inference.sd3_text_encoders import (
            TextEncoderModule,
        )

        torch_encoder_model = TextEncoderModule(
            args.batch_size,
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
