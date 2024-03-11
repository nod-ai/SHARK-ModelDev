import os
import subprocess
from datetime import datetime as dt
import torch

# Assuming BREAK_POS_F32, DTYPE_F32, BREAK_POS_F16, DTYPE_F16 are defined elsewhere
PATH_TO_SHARK_TURBINE=None
PATH_TO_JITTABLE=f"{PATH_TO_SHARK_TURBINE}/core/shark_turbine/aot/builtins/jittable.py"
HF_AUTH_KEY = None

def replace_values_in_file(path_to_file, new_break_pos, new_dtype, new_model):
    """
    Replaces BREAK_POS and DTYPE values in a file with user-specified values at the same line,
    preserving original indentation.

    Args:
        path_to_file (str): Path to the file to be modified.
        new_break_pos (int): New value for BREAK_POS.
        new_dtype (str): New value for DTYPE.

    Raises:
        FileNotFoundError: If the file is not found.
    """

    try:
        with open(path_to_file, "r") as file:
            lines = file.readlines()

        modified_lines = []
        for i, line in enumerate(lines):
            indentation = len(line) - len(line.lstrip())
            if "BREAK_POS = " in line:
                modified_lines.append(indentation * " " + f"BREAK_POS = {new_break_pos}\n")
            elif "DTYPE = " in line:
                modified_lines.append(indentation * " " + f"DTYPE = {new_dtype}\n")
            elif "MODEL = " in line:
                modified_lines.append(indentation * " " + f"MODEL = '{new_model}'\n")
            else:
                modified_lines.append(line)

        with open(path_to_file, "w") as file:
            file.writelines(modified_lines)

        print(f"Values replaced successfully in {path_to_file} with indentation preserved.")
    except FileNotFoundError:
        print(f"Error: File not found at {path_to_file}")


def create_next_test_dir(init_folder_name):
    i = 0
    while (os.path.isdir(f"{init_folder_name}_{i:02}")):
        i+=1
    folder_name = f"{init_folder_name}_{i:02}"
    os.mkdir(f"{folder_name}")
    print(f"dir created - {folder_name}")
    return folder_name


def write_subprocess_result_to_file(result, file_name):
    with open(file_name, "w") as f:
        print(result.stdout, file=f)
        print(result.stderr, file=f)


def unet(BREAK_POS_F16, BREAK_POS_F32):
    DTYPE_F32 = torch.float32
    DTYPE_F16 = torch.float16
    folder_name = create_next_test_dir(f"small_unet_graphs_rocm_{BREAK_POS_F16}")
    os.chdir(folder_name)

    # Replace values for FP32
    replace_values_in_file(PATH_TO_JITTABLE, BREAK_POS_F32, DTYPE_F32, "unet")

    start = dt.now()
    print(f"{start.strftime('%H:%M:%S.%f')} : F32 Start")
    # Run FP32 command
    command = f"""time python {PATH_TO_SHARK_TURBINE}/models/turbine_models/custom_models/sdxl_inference/unet_cpu_f32_torch.py --hf_auth_token={HF_AUTH_KEY} --compile_to=torch --external_weights=safetensors --external_weight_path={PATH_TO_SHARK_TURBINE}/stable_diffusion_xlv1p0_unet_fp32.safetensors --device=cpu --hf_model_name="stabilityai/stable-diffusion-xl-base-1.0" --iree_target_triple=x86_64-unknown-unknown-eabi-elf --max_length=64 --precision="fp32"
"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    write_subprocess_result_to_file(result, "report_run_f32.txt")
    end = dt.now()
    print(f"{end.strftime('%H:%M:%S.%f')} : F32 end\nElapsed{str(end-start)}")

    # Replace values for FP16
    replace_values_in_file(PATH_TO_JITTABLE, BREAK_POS_F16, DTYPE_F16, "unet")

    start = dt.now() 
    print(f"{start.strftime('%H:%M:%S.%f')} : F16 Start")
    # Run FP16 command
    command = f"""time python {PATH_TO_SHARK_TURBINE}/models/turbine_models/custom_models/sdxl_inference/unet.py --hf_auth_token={HF_AUTH_KEY} --compile_to=vmfb --external_weights=safetensors --external_weight_path={PATH_TO_SHARK_TURBINE}/stable_diffusion_xlv1p0_unet.safetensors --device=rocm --hf_model_name="stabilityai/stable-diffusion-xl-base-1.0" --iree_target_triple=gfx940 --max_length=64 --precision="fp16"
"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    write_subprocess_result_to_file(result, "report_run_f16.txt")
    end = dt.now()
    print(f"{end.strftime('%H:%M:%S.%f')} : F16 end\nElapsed {str(end-start)}")

    # Run final command and save output/error to report.txt
    command = f"""python {PATH_TO_SHARK_TURBINE}/models/turbine_models/custom_models/sdxl_inference/unet_runner.py --vmfb_path=stable_diffusion_xl_base_1_0_64_1024x1024_fp16_unet_rocm.vmfb --external_weight_path={PATH_TO_SHARK_TURBINE}/stable_diffusion_xlv1p0_unet.safetensors --compare_vs_torch --hf_auth_token={HF_AUTH_KEY} --device=rocm --precision=fp16 --max_length=64
"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    write_subprocess_result_to_file(result, "report.txt")

    os.chdir("../")


def vae(BREAK_POS_F16, BREAK_POS_F32):
    DTYPE_F32 = torch.float32
    DTYPE_F16 = torch.float16
    folder_name = create_next_test_dir(f"small_vae_graphs_rocm_{BREAK_POS_F16}")
    # folder_name = create_next_test_dir(f"/mnt/dcgpuval/<your folder>/small_vae_graphs_rocm_{BREAK_POS_F16}")
    os.chdir(folder_name)

    # Replace values for FP32
    replace_values_in_file(PATH_TO_JITTABLE, BREAK_POS_F32, DTYPE_F32, "vae")

    start = dt.now()
    print(f"{start.strftime('%H:%M:%S.%f')} : F32 Start")
    # Run FP32 command
    command = f"""time python {PATH_TO_SHARK_TURBINE}/models/turbine_models/custom_models/sdxl_inference/vae_cpu_f32_torch.py --compile_to=torch --external_weights=safetensors --external_weight_path={PATH_TO_SHARK_TURBINE}/stable_diffusion_xl_base_1_0_fp32_vae_decode.safetensors --device=cpu --hf_model_name="stabilityai/stable-diffusion-xl-base-1.0" --iree_target_triple=x86_64-unknown-unknown-eabi-elf --precision="fp32"
"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    write_subprocess_result_to_file(result, "report_run_f32.txt")
    end = dt.now()
    print(f"{end.strftime('%H:%M:%S.%f')} : F32 end\nElapsed {str(end-start)}")

    # Replace values for FP16
    replace_values_in_file(PATH_TO_JITTABLE, BREAK_POS_F16, DTYPE_F16, "vae")

    start = dt.now() 
    print(f"{start.strftime('%H:%M:%S.%f')} : F16 Start")
    # Run FP16 command
    command = f"""time python {PATH_TO_SHARK_TURBINE}/models/turbine_models/custom_models/sdxl_inference/vae.py --compile_to=vmfb --external_weights=safetensors --external_weight_path={PATH_TO_SHARK_TURBINE}/stable_diffusion_xl_base_1_0_vae_decode.safetensors --device=rocm --hf_model_name="stabilityai/stable-diffusion-xl-base-1.0" --iree_target_triple=gfx940 --precision="fp16"
"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    write_subprocess_result_to_file(result, "report_run_f16.txt")
    end = dt.now()
    print(f"{end.strftime('%H:%M:%S.%f')} : F16 end\nElapsed {str(end-start)}")

    # Run final command and save output/error to report.txt
    command = f"""python {PATH_TO_SHARK_TURBINE}/models/turbine_models/custom_models/sdxl_inference/vae_runner.py --vmfb_path=stable_diffusion_xl_base_1_0_1024x1024_fp16_vae_decode_rocm.vmfb --external_weight_path={PATH_TO_SHARK_TURBINE}/stable_diffusion_xl_base_1_0_vae_decode.safetensors --compare_vs_torch --device=rocm --precision=fp16
"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    write_subprocess_result_to_file(result, "report.txt")
    
    os.chdir("../")


if __name__ == "__main__":

    # unet(F16, F32)
    # transpose_3
    # unet(195, 192)
    # unet(191, 188)
    # unet(189, 186)
    # unet(187, 184)
    # unet(195, 192)

    # vae(36,36) #=> mm
    vae(33, 33) # transpose
    vae(41, 41) # mm
    vae(46, 46) # mm
    vae(43, 43) # add after mm
    vae(48, 48) # add after mm
    vae(38, 38) # add after mm 
