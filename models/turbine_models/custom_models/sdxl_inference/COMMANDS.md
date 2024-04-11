
# SHARK-Turbine SDXL CLI usage (ROCM)

## Pipeline (txt2img):

Note: These commands are generally for unix, and use `$WEIGHTS_DIR`, `$PIPELINE_DIR`, and `$TARGET_TRIPLE` in place of actual values. You can set these env variables or replace them in the commands as desired.

```shell
python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_compiled_pipeline.py --precision=fp16 --external_weights=irpa --device=rocm --rt_device=rocm --iree_target_triple=$TARGET_TRIPLE --scheduler_id=PNDM --num_inference_steps=30 --pipeline_dir=$PIPELINE_DIR --external_weights_dir=$WEIGHTS_DIR --attn_spec=default --compiled_pipeline

iree-benchmark-module \
 --module=$PWD/stable_diffusion_xl_base_1_0_64_fp16_prompt_encoder_rocm.vmfb \
 --parameters=model=$WEIGHTS_DIR/prompt_encoder.irpa \
 --module=$PWD/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_scheduled_unet_rocm.vmfb \
 --parameters=model=$WEIGHTS_DIR/unet.irpa \
 --module=$PWD/stable_diffusion_xl_base_1_0_1024x1024_fp16_vae_decode_rocm.vmfb \
 --parameters=model=$WEIGHTS_DIR/vae_decode.irpa \
 --module=$PWD/sdxl_pipeline_fp16_$TARGET_TRIPLE.vmfb \
 --function=tokens_to_image \
 --input=1x4x128x128xf16 \
 --input=1xf16 \
 --input=1x64xi64 \
 --input=1x64xi64 \
 --input=1x64xi64 \
 --input=1x64xi64 \
 --device_allocator=caching \
 --benchmark_repetitions=1 \
 --device=rocm
```
Note: you can either manually compile the pipeline vmfb from the .mlir in sdxl_inference, or by running the sdxl_scheduled_unet.py script.
The sdxl_compiled_pipeline script will do this for you, and you can switch between the segmented pipeline and the 'tokens->image' one-shot pipeline using `--compiled_pipeline` (if present, script will run the latter.)

## Scheduled UNet

```
# Import to MLIR:

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_scheduled_unet.py --precision=fp16 --external_weights=safetensors --device=rocm --compile_to=mlir --external_weight_path=$WEIGHTS_DIR/unet.safetensors

# Compile to VMFB (MLIR not necessary here but this is faster if you are compiling more than once):

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_scheduled_unet.py --precision=fp16 --external_weights=safetensors --device=rocm --compile_to=vmfb --iree_target_triple=$TARGET_TRIPLE --input_mlir=$PWD/stable_diffusion_xl_base_1_0_PNDM_64_1024x1024_fp16_unet_30.mlir

# Test numerics (validate against pytorch cpu):

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_scheduled_unet_runner.py --compare_vs_torch --precision=fp16 --device=rocm --external_weight_path=$WEIGHTS_DIR/scheduled_unet.irpa --max_length=64 --pipeline_vmfb_path=./sdxl_pipeline_fp16_$TARGET_TRIPLE.vmfb --vmfb_path=$PWD/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_scheduled_unet_rocm.vmfb

# Benchmark with IREE CLI:

iree-benchmark-module --benchmark_repetitions=5   --device=rocm --module=$PWD/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_scheduled_unet_rocm.vmfb --parameters=model=$WEIGHTS_DIR/scheduled_unet.irpa   --function=run_forward     --input=1x4x128x128xf16 --input=2x64x2048xf16  --input=2x1280xf16  --input=2x6xf16 --input=1xf16 --input=1xi64 --device_allocator=caching

iree-benchmark-module --benchmark_repetitions=5   --device=rocm --module=$PWD/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_scheduled_unet_rocm.vmfb --module=$PWD/sdxl_pipeline_fp16_$TARGET_TRIPLE.vmfb --parameters=model=$WEIGHTS_DIR/scheduled_unet.irpa   --function=run_forward     --input=1x4x128x128xf16 --input=2x64x2048xf16  --input=2x1280xf16  --input=2x6xf16 --input=1xf16 --input=1xi64 --device_allocator=caching
```

## UNet

```
# Import to MLIR:

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/unet.py --precision=fp16 --external_weights=safetensors --device=rocm --compile_to=mlir

# Compile to VMFB (MLIR not necessary here but this is faster if you are compiling more than once):

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/unet.py --precision=fp16 --external_weights=safetensors --device=rocm --compile_to=vmfb --iree_target_triple=$TARGET_TRIPLE --input_mlir=$PWD/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet.mlir

# Convert weights to IREE parameter archive format:

iree-convert-parameters --parameters=$WEIGHTS_DIR/unet.safetensors --output=$WEIGHTS_DIR/scheduled_unet.irpa

# Test numerics (validate against pytorch cpu):

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/unet_runner.py --compare_vs_torch --precision=fp16 --device=rocm --external_weight_path=$WEIGHTS_DIR/scheduled_unet.irpa --max_length=64 --vmfb_path=$PWD/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet_rocm.vmfb

# Benchmark with IREE CLI:

iree-benchmark-module --benchmark_repetitions=5   --device=rocm --module=$PWD/stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet_rocm.vmfb --parameters=model=$WEIGHTS_DIR/scheduled_unet.irpa   --function=main   --input=1x4x128x128xf16 --input=1xi64 --input=2x64x2048xf16 --input=2x1280xf16  --input=2x6xf16 --input=1xf16  --device_allocator=caching
```

## CLIP

```
# Import to MLIR:

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_prompt_encoder.py --precision=fp16 --external_weights=safetensors --device=rocm --rt_device=rocm --compile_to=mlir --iree_target_triple=$TARGET_TRIPLE --external_weight_path=$WEIGHTS_DIR/prompt_encoder.safetensors

# Compile to VMFB (MLIR not necessary here but this is faster if you are compiling more than once):

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_prompt_encoder.py --precision=fp16 --external_weights=safetensors --device=rocm --rt_device=rocm --compile_to=vmfb --iree_target_triple=$TARGET_TRIPLE --input_mlir=$PWD/stable_diffusion_xl_base_1_0_64_fp16_prompt_encoder.mlir

# Convert weights to IREE parameter archive format:

iree-convert-parameters --parameters=$WEIGHTS_DIR/prompt_encoder.safetensors --output=$WEIGHTS_DIR/prompt_encoder.irpa

# Test numerics (validate against pytorch cpu):

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/sdxl_prompt_encoder_runner.py --compare_vs_torch --precision=fp16 --device=rocm --external_weight_path=$WEIGHTS_DIR/prompt_encoder.irpa --max_length=64 --vmfb_path=$PWD/stable_diffusion_xl_base_1_0_64_fp16_prompt_encoder_rocm.vmfb

# Benchmark with IREE CLI:

iree-benchmark-module --benchmark_repetitions=5   --device=rocm --module=$PWD/stable_diffusion_xl_base_1_0_64_fp16_prompt_encoder_rocm.vmfb --parameters=model=$WEIGHTS_DIR/prompt_encoder.irpa   --function=encode_prompts     --input=1x64xi64  --input=1x64xi64  --input=1x64xi64  --input=1x64xi64  --device_allocator=caching
```


## VAE

```
# Import to MLIR:

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/vae.py --precision=fp16 --external_weights=safetensors --device=rocm --compile_to=mlir --iree_target_triple=$TARGET_TRIPLE --external_weight_path=$WEIGHTS_DIR/vae_decode.safetensors

# Compile to VMFB (MLIR not necessary here but this is faster if you are compiling more than once):

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/vae.py --precision=fp16 --external_weights=safetensors --device=rocm --compile_to=vmfb --iree_target_triple=$TARGET_TRIPLE --input_mlir=$PWD/stable_diffusion_xl_base_1_0_1024x1024_fp16_vae_decode.mlir

# Convert weights to IREE parameter archive format:

iree-convert-parameters --parameters=$WEIGHTS_DIR/vae_decode.safetensors --output=$WEIGHTS_DIR/vae_decode.irpa

# Test numerics (validate against pytorch cpu):

python /home/eagarvey/sdxl/SHARK-Turbine/models/turbine_models/custom_models/sdxl_inference/vae_runner.py --precision=fp16 --external_weights=irpa --device=rocm --iree_target_triple=$TARGET_TRIPLE --vmfb_path=$PWD/stable_diffusion_xl_base_1_0_1024x1024_fp16_vae_decode_rocm.vmfb --external_weight_path=$WEIGHTS_DIR/vae_decode.irpa --compare_vs_torch

# Benchmark with IREE CLI:

iree-benchmark-module --benchmark_repetitions=5 --module=$PWD/stable_diffusion_xl_base_1_0_1024x1024_fp16_vae_decode_rocm.vmfb --parameters=model=$WEIGHTS_DIR/vae_decode.irpa --device=rocm --input=1x4x128x128xf16 --device-allocator=caching --function=main
```
