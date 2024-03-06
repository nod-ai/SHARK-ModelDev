def pytest_addoption(parser):
    # Huggingface Options
    parser.addoption("--hf_auth_token", action="store", default=None)
    parser.addoption(
        "--hf_model_name",
        action="store",
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.addoption("--scheduler_id", action="store", default="PNDM")
    # Inference Options
    parser.addoption(
        "--prompt",
        action="store",
        default="a photograph of an astronaut riding a horse",
    )
    parser.addoption(
        "--negative_prompt",
        action="store",
        default="blurry, unsaturated, watermark, noisy, grainy, out of focus",
    )
    parser.addoption("--num_inference_steps", type=int, action="store", default=30)
    parser.addoption("--guidance_scale", type=float, action="store", default=7.5)
    parser.addoption("--seed", type=float, action="store", default=0.0)
    parser.addoption("--vmfb_path", action="store", default="")
    parser.addoption("--external_weight_path", action="store", default="")
    parser.addoption("--external_weight_dir", action="store", default="")
    parser.addoption("--external_weight_file", action="store", default="")
    parser.addoption("--pipeline_dir", action="store", default="")
    # Modelling Options
    parser.addoption("--batch_size", type=int, action="store", default=1)
    parser.addoption("--height", type=int, action="store", default=1024)
    parser.addoption("--width", type=int, action="store", default=1024)
    parser.addoption("--precision", action="store", default="fp16")
    parser.addoption("--max_length", type=int, action="store", default=64)
    parser.addoption("--run_vmfb", action="store", default=True)
    # General Options
    parser.addoption("--compile_to", action="store", default=None)
    parser.addoption("--external_weights", action="store", default="safetensors")
    parser.addoption("--decomp_attn", action="store_true", default=False)
    # Compiler Options
    parser.addoption("--device", action="store", default="cpu")
    parser.addoption("--rt_device", action="store", default="local-task")
    parser.addoption(
        "--iree_target_triple", type=str, action="store", default="x86_64-linux-gnu"
    )
    parser.addoption("--ireec_flags", action="store", default="")
    parser.addoption("--attn_flags", action="store", default="")
    # Test Options
    parser.addoption("--in_channels", type=int, action="store", default=4)
    parser.addoption("--benchmark", action="store_true", default=False)
    parser.addoption("--tracy_profile", action="store_true", default=False)
