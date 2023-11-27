# Instructions

Clone and install SHARK-Turbine
```
git clone git@github.com:nod-ai/SHARK-Turbine.git
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

To generate the vmfb for the benchmark
```
python python/turbine_models/custom_models/stateless_llama.py --compile_to=vmfb --hf_auth_token=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 --external_weights="safetensors" --quantization="int4" --precision="f16"
```
By default the model will be saved as `Llama_2_7b_chat_hf.vmfb`.


# E2E Benchmarking

To run the benchmark, use this command:

```
python python/turbine_models/custom_models/stateless_llama.py --run_benchmark --hf_auth_token=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 --vmfb_path=Llama_2_7b_chat_hf.vmfb --external_weight_file=Llama_2_7b_chat_hf_f16_int4.safetensors --benchmark_steps=10
```
