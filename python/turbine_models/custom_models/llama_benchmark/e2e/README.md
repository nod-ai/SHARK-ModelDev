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

## Compiling LLMs
Note: Make sure to replace "your_token" with your actual hf_auth_token for all the commands.

Now, you can generate the quantized weight file with
```
python python/turbine_models/gen_external_params/gen_external_params.py --hf_auth_token=your_token
```
The model weights will then be saved in the current directory as `Llama_2_7b_chat_hf_f16_int4.safetensors`.

To compile to vmfb for llama
```
python python/turbine_models/custom_models/stateless_llama.py --compile_to=vmfb --hf_auth_token=your_token --external_weights="safetensors" --quantization="int4" --precision="f16"
```
By default the vmfb will be saved as `Llama_2_7b_chat_hf.vmfb`.

##  Benchmarking LLMs e2e
To run benchmark with the default benchmark dataset just run:
```
python python/turbine_models/custom_models/llama-benchmark/e2e/llm_e2e_benchmark.py --vmfb_path=/path/to/Llama_2_7b_chat_hf.vmfb --external_weight_path=Llama_2_7b_chat_hf_f16_int4.safetensors --device=vulkan hf_auth_token=your_hf_token
```
You can specify a path to dataset using: `--benchmark_dataset_path=/path/to/dataset.json`
You can specify where to store the result path using: `--benchmark_output_path=/path/to/output.json`

## Benchmarking Dataset

To setup a dataset json you'd need a json file with a list of entry(s) containing these attributes:
1. id : number identifying example (int)
2. system_prompt : System prompt to align LLM (str)
3. user_prompt : Query example from user (str)
4. num_iterations : number of times to run/benchmark the particular example (int)
5. num_tokens_to_generate : how many tokens do we want to generate for the example (int)

Here is a sample:
```json
[
    {"id" : 0,
    "system_prompt": "<s>[INST] <<SYS>>\nBe concise. You are a helpful, respectful and honest assistant. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n <</SYS>>\n\n",
    "user_prompt" : "what is the capital of canada?",
    "num_iterations": 8,
    "num_tokens_to_generate": 20}
]
```

The default dataset in `benchmark_prompts.json` contains example that SHARK-1.0 traditionally measures. Additionally, we also added some data common in MLPerf which uses some data from open-orca. In the future, we should add more of the data from open-orca to run benchmarks with. 

## Benchmarking Output

The output json will have similar attributes with an addition of the results/measured benchmarks. Hence it will have these additional attributes:
1. prefill_tokens : number of tokens ran during the prefill stage (int)
2. prefill_speed(tok/s) : Number of tokens for initial input / time to complete prefill (float)
3. decoded_tokens : number of tokens decoded during decode stage. (int)
4. decode_speed(tok/s) : Average speed of decoding per token for this example, averaged over the number of iterations. (float)
