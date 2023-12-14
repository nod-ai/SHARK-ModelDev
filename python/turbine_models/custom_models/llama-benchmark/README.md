# Instructions

Clone and install SHARK-Turbine
```
git clone https://github.com/nod-ai/SHARK-Turbine.git
cd SHARK-Turbine
python -m venv turbine_venv && source turbine_venv/bin/activate

pip install --upgrade -r requirements.txt
pip install --upgrade -e .[torch-cpu-nightly,testing]
pip install --upgrade -r turbine-models-requirements.txt
```

Note: Make sure to replace "your_token" with your actual hf_auth_token for all the commands.

Now, you can generate the quantized weight file with
```
python python/turbine_models/gen_external_params/gen_external_params.py
```
The model weights will then be saved in the current directory as `Llama_2_7b_chat_hf_f16_int4.safetensors`.

To compile to vmfb for llama
```
python python/turbine_models/custom_models/stateless_llama.py --compile_to=vmfb --hf_auth_token=your_token --external_weights="safetensors" --quantization="int4" --precision="f16"
```
By default the vmfb will be saved as `Llama_2_7b_chat_hf.vmfb`.

There are two options provided for benchmarking:

1) Benchmarking the first and second vicuna (run_initialize and run_forward)
2) Only benchmarking the second vicuna (run_forward) for more accurate tok/s

To compile to vmfb for benchmark option 1:
```
python python/turbine_models/custom_models/llama-benchmark/benchmark_module.py --benchmark_mlir_path=./python/turbine_models/custom_models/llama-benchmark/benchmark.mlir
```
By default the vmfb will be saved as `benchmark.vmfb`.

To compile to vmfb for benchmark option 2:
```
python python/turbine_models/custom_models/llama-benchmark/benchmark_module.py --benchmark_mlir_path=./python/turbine_models/custom_models/llama-benchmark/benchmark_forward.mlir
```
By default the vmfb will be saved as `benchmark.vmfb`.


# Benchmarking

Set the number of times second vicuna is run (# of tokens to benchmark) using the steps argument in following command.

To run the benchmark, use this command for option 1 (first and second vicuna):

```
python python/turbine_models/custom_models/llama-benchmark/stateless_llama_benchmark.py --hf_auth_token=your_token --benchmark_vmfb_path=benchmark.vmfb --llama_vmfb_path=Llama_2_7b_chat_hf.vmfb --external_weight_file=Llama_2_7b_chat_hf_f16_int4.safetensors --steps=10
```

To run the benchmark, use this command for option 2 (only run_forward):

```
python python/turbine_models/custom_models/llama-benchmark/stateless_llama_benchmark.py --run_forward_only_benchmark --hf_auth_token=your_token --benchmark_vmfb_path=benchmark.vmfb --llama_vmfb_path=Llama_2_7b_chat_hf.vmfb --external_weight_file=Llama_2_7b_chat_hf_f16_int4.safetensors --steps=10
```