import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_cmd (cmd, pipeline, flags):
    print(bcolors.BOLD + bcolors.OKGREEN)
    print (cmd, pipeline)
    for f in flags:
        print("\t", f)
    print(bcolors.ENDC)

cmd = "python"
pipeline = "models/turbine_models/custom_models/sd_inference/sd_pipeline.py"
prompt = "Photo of a ultra realistic sailing ship, dramatic light, pale sunrise, cinematic lighting, battered, low angle, trending on artstation, 4k, hyper realistic, focused, extreme details, unreal engine 5, cinematic, masterpiece, art by studio ghibli, intricate artwork by john william turner"
height = 512
width=512
mmdit_onnx_model_path = "C:/Users/chiz/work/sd3/mmdit/fp32/mmdit_optimized.onnx"
flags = [
        "--hf_model_name=stabilityai/stable-diffusion-3-medium-diffusers",
        f"--height={height}",
        f"--width={width}",
        "--clip_device=local-task",
        "--clip_precision=fp16",
        "--clip_target=znver4",
        "--clip_decomp_attn",
        "--mmdit_precision=fp16",
        "--mmdit_device=rocm-legacy://0",
        "--mmdit_target=gfx1150",
        '''--mmdit_flags="masked_attention" ''',
        "--run_onnx_mmdit",
        f'''--mmdit_onnx_model_path="{mmdit_onnx_model_path}" ''',
        "--vae_device=rocm-legacy://0",
        "--vae_precision=fp16",
        "--vae_target=gfx1150",
        '''--vae_flags="masked_attention" ''',
        "--external_weights=safetensors", 
        "--num_inference_steps=28",
        "--verbose",
        f'''--prompt="{prompt}" '''
        ]

print_cmd(cmd, pipeline, flags)

final_cmd = ' '.join([cmd, pipeline]+flags)
os.system(final_cmd)