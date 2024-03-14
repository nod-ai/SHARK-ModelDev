# Stable Diffusion Commands

## Run and benchmark the entire SDXL pipeline on MI300
 - note: the command below is specifically for use on the ppac-pla-s22-35 instance. you may need to tweak paths accordingly.
 - follow "setup repository" in the next section
 - optional: set HF_HOME to save dl time/ disk usage
```
export HF_HOME=/mnt/dcgpuval/huggingface/     #ppac
export HF_HOME=/data/huggingface-cache        #banff
```
 - make sure you have ROCM working with IREE, check `iree-run-module --dump_devices`
 - make a file called "mfma_spec.mlir" and drop in the contents of the TD script https://github.com/nod-ai/2024-q1-sdxl-sprint/tree/main/specs.

### Newest pipeline command, weights (as of [SHARK-Turbine@ean-sd-fp16:824f43e](https://github.com/nod-ai/SHARK-Turbine/commit/824f43e83a53d49307ddfe0b829da22c69ac2ddd)):

```
python SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_compiled_pipeline.py --precision=fp16 --external_weights=safetensors --device=rocm --rt_device=rocm --iree_target_triple=gfx942 --scheduler_id=PNDM --num_inference_steps=30 --pipeline_dir=./sdxl_fp16_1024x1024_gfx942/ --external_weights_dir=./weights_fp16/
```

Note: the following "prompt_encoder_f16.irpa" contains weights for both clip1 and clip2.
The pipeline script will look for these filenames in the specified "external_weights_dir" under "prompt_encoder.irpa", "vae_decode.irpa", "scheduled_unet.irpa".
It's not ideal in current state, but will be smoothed out now that general pipeline structure and file management needs are stable.
 - [prompt_encoder_f16.irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/SDXL_weights_fp16/prompt_encoder_fp16.irpa)
 - [scheduled_unet_f16.irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/SDXL_weights_fp16/scheduled_unet_f16.irpa)
 - [vae_decode_f16.irpa](https://sharkpublic.blob.core.windows.net/sharkpublic/SDXL/SDXL_weights_fp16/vae_encode_fp16.irpa)
