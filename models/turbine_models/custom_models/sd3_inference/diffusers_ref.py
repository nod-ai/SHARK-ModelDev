from diffusers import StableDiffusion3Pipeline
import torch
from datetime import datetime as dt


def run_diffusers_cpu(
    hf_model_name,
    prompt,
    negative_prompt,
    guidance_scale,
    seed,
    height,
    width,
    num_inference_steps,
):
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        hf_model_name, torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")
    generator = torch.Generator().manual_seed(int(seed))

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    image.save(f"diffusers_reference_output_{timestamp}.png")


if __name__ == "__main__":
    from turbine_models.custom_models.sd_inference.sd_cmd_opts import args

    run_diffusers_cpu(
        args.hf_model_name,
        args.prompt,
        args.negative_prompt,
        args.guidance_scale,
        args.seed,
        args.height,
        args.width,
        args.num_inference_steps,
    )
