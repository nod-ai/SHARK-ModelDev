# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
import torch._dynamo as dynamo
from torch._export import dynamic_dim
from torch._export.constraints import constrain_as_size, constrain_as_value
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from PIL import Image
from diffusers import LMSDiscreteScheduler

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

class VaeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae"
        )

    def forward(self, input):
        x = self.vae.encode(input, return_dict=False)[0]
        return x

class UnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet"
        )

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet.forward(sample, timestep, encoder_hidden_states)[0]


def load_models():
    vae = VaeModel()
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer"
    )
    text_encoder_model = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder"
    )
    unet = UnetModel()

    vae.eval()
    unet.eval()

    return vae, tokenizer, text_encoder_model, unet


def perform_inference(vae, tokenizer, text_encoder_model, unet, prompt):
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)

    height = 512
    width = 512
    num_inference_steps = 5
    guidance_scale = 7.5
    generator = torch.manual_seed(0)
    batch_size = len(prompt)

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder_model(text_input.input_ids)[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder_model(uncond_input.input_ids)[0]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.unet.in_channels, height // 8, width // 8),
        generator=generator,
    )

    scheduler.set_timesteps(num_inference_steps)

    latents = latents * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images[0]


if __name__ == "__main__":
    vae, tokenizer, text_encoder_model, unet = load_models()
    prompt = ["a photograph of an astronaut riding a horse"]
    generated_image = perform_inference(vae, tokenizer, text_encoder_model, unet, prompt)
    generated_image.save('output.jpg')