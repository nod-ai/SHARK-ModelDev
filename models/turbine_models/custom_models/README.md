# Instructions

Clone and install SHARK-Turbine
```
git clone https://github.com/nod-ai/SHARK-Turbine.git
cd SHARK-Turbine
python -m venv turbine_venv && source turbine_venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cpu \
    -r core/pytorch-cpu-requirements.txt
pip install --upgrade -r core/requirements.txt
pip install -e core
pip install -e models
```

## Compiling LLMs
Note: Make sure to replace "your_token" with your actual hf_auth_token for all the commands.

Now, you can generate the quantized weight file with
```
python models/turbine_models/gen_external_params/gen_external_params.py --hf_auth_token=your_token
```
The model weights will then be saved in the current directory as `Llama_2_7b_chat_hf_f16_int4.safetensors`.

To compile to vmfb for llama
```
python models/turbine_models/custom_models/stateless_llama.py --compile_to=vmfb --hf_auth_token=your_token --external_weights="safetensors" --quantization="int4" --precision="f16"
```
By default the vmfb will be saved as `Llama_2_7b_chat_hf.vmfb`.

##  Running LLMs
There are two ways of running LLMs:

1) Single run with predefined prompt to validate correctness.
```
python models/turbine_models/custom_models/llm_runner.py --vmfb_path=/path/to/Llama_2_7b_chat_hf.vmfb --external_weight_path=Llama_2_7b_chat_hf_f16_int4.safetensors --device=vulkan hf_auth_token=your_hf_token
```
2) Interactive CLI chat mode. (just add a --chat_mode flag)
```
python models/turbine_models/custom_models/llm_runner.py --vmfb_path=/path/to/Llama_2_7b_chat_hf.vmfb --external_weight_path=Llama_2_7b_chat_hf_f16_int4.safetensors --device=vulkan hf_auth_token=your_hf_token --chat_mode
```
