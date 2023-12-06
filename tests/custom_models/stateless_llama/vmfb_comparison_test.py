import pytest
import subprocess
import os
import glob

def delete_files(pattern):
    # Delete files matching the given pattern
    for file in glob.glob(pattern):
        os.remove(file)

@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown():
    # Setup: Delete existing files
    delete_files('*.safetensors')
    delete_files('*.ggml')
    delete_files('*.vmfb')
    delete_files('*.mlir')
    # Yield to the test execution
    yield
    # Teardown: Delete files after tests
    delete_files('*.safetensors')
    delete_files('*.ggml')
    delete_files('*.vmfb')
    delete_files('*.mlir')

@pytest.fixture
def setup_environment():
    # Change to the SHARK-Turbine directory
    # os.chdir(os.path.expanduser('~/SHARK-Turbine'))
    # Ensure that any failure in the commands causes the test to stop
    # subprocess.run('set -e', shell=True, check=True)
    pass

def run_command(command):
    # Run the command and check for successful execution
    subprocess.run(command, shell=True, check=True)

def test_generate_vmfb(setup_environment):
    command = 'python python/turbine_models/custom_models/stateless_llama.py --compile_to=vmfb --hf_model_name="llSourcell/medllama2_7b" --precision=f16 --quantization=int4  --external_weights=safetensors'
    run_command(command)

def test_generate_quantized_safetensors(setup_environment):
    command = 'python python/turbine_models/gen_external_params/gen_external_params.py --hf_model_name="llSourcell/medllama2_7b" --precision=f16 --quantization=int4'
    run_command(command)

def test_run_vmfb_vs_torch_model(setup_environment):
    command = 'python python/turbine_models/custom_models/stateless_llama.py --run_vmfb --hf_model_name="llSourcell/medllama2_7b" --vmfb_path=medllama2_7b.vmfb --external_weight_file=medllama2_7b_f16_int4.safetensors'
    run_command(command)
