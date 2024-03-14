python ..\models\turbine_models\custom_models\sdxl_inference\unet.py --compile_to=mlir --external_weights=safetensors --max_length=64 --precision="fp16" --iree_target_triple=x86_64-linux-gnu --height=1024 --width=1024 --external_weight_path=./stable_diffusion_xl_base_1_0_fp16_unet.safetensors


python ..\models\turbine_models\custom_models\sdxl_inference\clip.py --compile_to=mlir --external_weights=safetensors --max_length=64 --precision="fp16" --iree_target_triple=x86_64-linux-gnu --external_weight_path=./stable_diffusion_xl_base_1_0_fp16_clip.safetensors


python ..\models\turbine_models\custom_models\sdxl_inference\vae.py --compile_to=mlir --external_weights=safetensors --precision="fp16" --vae_variant=decode --iree_target_triple=x86_64-linux-gnu --height=1024 --width=1024 --external_weight_path=./stable_diffusion_xl_base_1_0_fp16_vae_decode.safetensors



python ..\models\turbine_models\custom_models\sdxl_inference\unet.py --compile_to=mlir --external_weights=safetensors --precision="fp32" --max_length=64 --iree_target_triple=x86_64-linux-gnu --height=1024 --width=1024 --external_weight_path=./stable_diffusion_xl_base_1_0_fp32_unet.safetensors


python ..\models\turbine_models\custom_models\sdxl_inference\clip.py --compile_to=mlir --external_weights=safetensors --precision="fp32" --max_length=64 --iree_target_triple=x86_64-linux-gnu --external_weight_path=./stable_diffusion_xl_base_1_0_fp32_clip.safetensors


python ..\models\turbine_models\custom_models\sdxl_inference\vae.py --compile_to=mlir --external_weights=safetensors --precision="fp32" --vae_variant=decode --iree_target_triple=x86_64-linux-gnu --height=1024 --width=1024 --external_weight_path=./stable_diffusion_xl_base_1_0_fp32_vae_decode.safetensors


python ..\models\turbine_models\custom_models\sdxl_inference\sdxl_prompt_encoder.py --compile_to=mlir --external_weights=safetensors --precision="fp32" --max_length=64 --iree_target_triple=x86_64-linux-gnu --external_weight_path=./stable_diffusion_xl_base_1_0_fp32_prompt_encoder.safetensors