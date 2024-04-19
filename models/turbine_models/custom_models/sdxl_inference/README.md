# Stable Diffusion XL with SHARK-Turbine

## Support

Following is a table that shows current status of turbine SDXL inference support for a few AMDGPU targets. This is not an exhaustive list of supported targets.

| Target Chip | Attention Decomposed? | CLIP          | UNet                           | VAE Decode                     | Txt2Img        |
|-------------|-----------------------|---------------|--------------------------------|--------------------------------|----------------|
| gfx1100     | Yes                   | 💚 | 💛 (numerics with vector distribution)| 💚                  | 💚  |
|             | No                    |               | 💔 (Attn lowering) | 💔 (Attn lowering) | 💔 |
| gfx90a      | Yes                   | 💚 | 💚                  | 💚                  | 💚  |
|             | No                    |               | 💔 (Shared Memory) | 💚                  | 💔 |
| gfx942      | Yes                   | 💚 | 💚                  | 💚                  | 💚  |
|             | No                    |               | 💚                  | 💚                  | 💚  |

## Setup SHARK-Turbine for importing and running the SDXL pipeline or submodels.

Linux:
```shell
python -m venv turbine_venv
source turbine_venv/bin/activate
python -m pip install --upgrade pip
pip install -r core/pytorch-cpu-requirements.txt
pip install --pre --upgrade -r core/requirements.txt
pip install --pre -e core
pip install --pre --upgrade -e models -r models/requirements.txt
```

Windows:
```shell
python -m venv turbine_venv
turbine_venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r core/pytorch-cpu-requirements.txt
pip install --pre --upgrade -r core/requirements.txt
pip install --pre -e core
pip install --pre --upgrade -e models -r models/requirements.txt
```

## Run tests
ROCM:
```
pytest models/turbine_models/tests/sdxl_test.py --device=rocm --rt_device=<hip/rocm> --iree_target_triple=gfx<ID> --external_weights=safetensors
```

CPU:
```
pytest models/turbine_models/tests/sdxl_test.py --device=cpu --rt_device=local-task --iree_target_triple=x86-64_linux_gnu --external_weights=safetensors --precision=fp32
```

## Run image generation pipeline

ROCM:
```
python models\turbine_models\custom_models\sdxl_inference\sdxl_compiled_pipeline.py --iree_target_triple=gfx1100 --device=rocm --rt_device=hip --external_weights=safetensors
```
For mfma-capable hardware, use `--attn_spec=default` to lower attention ops to MFMA instructions.

CPU:
```
pytest models/turbine_models/tests/sdxl_test.py --device=cpu --rt_device=local-task --iree_target_triple=x86-64_linux_gnu --external_weights=safetensors --precision=fp32
```

## Shared CLI options
 - `--iree_target_triple`: use gfx1100 for 7900xt, gfx90a for MI210/MI250, gfx940 for MI300A, gfx942 for MI300X. For CPU, use x86_64-linux-gnu if you aren't sure. On Vulkan, this is something like `rdna3-7900-windows`.
 - `--rt_device`: if using pip install, `hip` will work correctly, but `rocm` will not. Source builds of IREE can support rocm with the `-DIREE_HAL_DRIVER_ROCM=ON -DIREE_EXTERNAL_HAL_DRIVERS="rocm"`, but that option is soon to be deprecated in favor of the HIP driver.
 - `--compiled_pipeline`: run one-shot SDXL in a MLIR wrapper, removing model glue from python execution layer
 - `--pipeline_dir`: directory in which to save or look for .vmfb files.
 - `--external_weights_dir`: directory in which to save or look for weights.
 - `--ireec_flags`: extra ireec flags to use for _all_ submodels.
 - `--unet_flags / --vae_flags / --clip_flags`: extra ireec flags for individual submodels.
 - `--precision`: fp16 or fp32. Default is fp16 and you should only use fp32 for cpu.
 - `--num_inference_steps`: (default 30) number of unet iterations to run.
 - `--batch_count`: Not compatible with `--compiled_pipeline`. Uses the same clip output to generate a set of images in a batch, with different image latents.
 - `--prompt / --negative_prompt`: prompts for stable diffusion inference


Note: the following "prompt_encoder_f16.irpa" contains weights for both clip1 and clip2.
The pipeline script will look for these filenames in the specified "external_weights_dir" under "prompt_encoder.irpa", "vae_decode.irpa", "scheduled_unet.irpa".
It's not ideal in current state, but will be smoothed out now that general pipeline structure and file management needs are stable.
 - [prompt_encoder_f16.irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/SDXL_weights_fp16/prompt_encoder_fp16.irpa)
 - [scheduled_unet_f16.irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/SDXL_weights_fp16/scheduled_unet_f16.irpa)
 - [vae_decode_f16.irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/SDXL_weights_fp16/vae_encode_fp16.irpa)
