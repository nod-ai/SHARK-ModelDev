# StreamingLLM

StreamingLLM is based on the paper *"Efficient Streaming Language Models with Attention Sinks"* by  Xiao et al from the MIT Han Lab.  Here is the original [[paper](http://arxiv.org/abs/2309.17453)] and [[code](https://github.com/mit-han-lab/streaming-llm)].

The modify_llama.py code is highly inspired by the modify_llama.py code in the original repo, but tweaked to work with ToM HuggingFace and compilable through Turbine.

The work introduces sink attention which in short is a combination of a fixed starting few sequence attention along with a sliding window attention. This is beneficial for these reasons:

1) Generate infinitely long context.
2) Maintain memory under certain threshold (controlled by window_length)


## Compiling LLMs with StreamingLLM

Just need to add an extra `--streaming_llm` flag when you call stateless_llama when generating your vmfb. For example:
```
python python/turbine_models/custom_models/stateless_llama.py --compile_to=vmfb --hf_auth_token=your_token --external_weights="safetensors" --quantization="int4" --precision="f16" --streaming_llm
```

By default the vmfb will still be saved as `Llama_2_7b_chat_hf.vmfb`.

## Running LLMs with StreamingLLM

Similar to compiling, just need to add an extra `--streaming_llm` flag when you call llm_runner.py. For example:
```
python python/turbine_models/custom_models/llm_runner.py --vmfb_path=/path/to/Llama_2_7b_chat_hf.vmfb --external_weight_path=Llama_2_7b_chat_hf_f16_int4.safetensors --device=vulkan hf_auth_token=your_hf_token --chat_mode --streaming_llm=true
```

## Future Work:
- [ ] Make window size configurable through python, everything is there but we'd need to initialize with a default value which would only be possible after we let `_create_initial_value` to take in initial value from GlobalAttribute somewhere [here](https://github.com/nod-ai/SHARK-Turbine/blob/18e8a4100b61adfd9425dd32f780dc5f90017813/python/shark_turbine/aot/support/ir_utils.py#L284-L316) . 
- [ ] Get flow.move to enable overlap of sliding window and src of data. (Currently need to evict when it's at least 2x size of window) For example by default our streamingLLM window_size is 256, so we evict at ~600(slightly more than 2x for safety) token.
- [ ] Introduce Rerotation of RoPE to as seen [here](https://github.com/huggingface/transformers/blob/c2d283a64a7f33547952e3eb0fa6533fc375bcdd/src/transformers/cache_utils.py#L213-L218) to remove invasive modification of LlamaAttention module for streamingLLM.