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

Now, you can generate the quantized weight file with
```
python python/turbine_models/gen_external_params/gen_external_params.py --hf_auth_token=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
```
The model weights will then be saved in the current directory as `Llama_2_7b_chat_hf_f16_int4.safetensors`.

To compile to vmfb for llama
```
python python/turbine_models/custom_models/stateless_llama.py --compile_to=vmfb --hf_auth_token=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 --external_weights="safetensors" --quantization="int4" --precision="f16"
```
By default the vmfb will be saved as `Llama_2_7b_chat_hf.vmfb`.

To compile to vmfb for benchmark
```
python python/turbine_models/custom_models/llama-benchmark/benchmark_module.py --benchmark_mlir_path=./python/turbine_models/custom_models/llama-benchmark/benchmark.mlir
```
By default the vmfb will be saved as `benchmark.vmfb`.


# Benchmarking

Set the number of times second vicuna is run (# of tokens to benchmark) using the steps argument in following command.

To run the benchmark, use this command:

```
python python/turbine_models/custom_models/llama-benchmark/stateless_llama_benchmark.py --hf_auth_token=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 --benchmark_vmfb_path=benchmark.vmfb --llama_vmfb_path=Llama_2_7b_chat_hf.vmfb --external_weight_file=Llama_2_7b_chat_hf_f16_int4.safetensors --steps=10
```
