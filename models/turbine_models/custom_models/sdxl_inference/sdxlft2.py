# Install the required libs
# pip install -U git+https://github.com/huggingface/diffusers.git
# pip install accelerate transformers ftfy

# sdxl copy

import logging
import torch
from turbine_models.custom_models.sdxl_inference.vae import VaeModel
from turbine_models.custom_models.sdxl_inference.unet import UnetModel
from turbine_models.custom_models.sdxl_inference.sdxl_prompt_encoder import PromptEncoderModule


import iree.runtime as ireert
from iree.compiler.ir import Context
from turbine_models.custom_models.sd_inference import utils
from turbine_models.utils.sdxl_benchmark import run_benchmark
from turbine_models.model_runner import vmfbRunner
from transformers import CLIPTokenizer
from shark_turbine.aot import *
from shark_turbine.dynamo.passes import (
    DEFAULT_DECOMPOSITIONS,
)

import unittest
from PIL import Image
import os
import numpy as np
import time
from datetime import datetime as dt

from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args
args.placeholder_token = "<pokemon>"
args.initializer_token = "pokemon"
args.use_torchdynamo = True
args.what_to_teach = "style"
args.training_steps = 2000
args.train_batch_size = 1
args.save_steps = 250

# Import required libraries
import argparse
import itertools
import math
import os
from typing import List
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
import logging

import torch_mlir
from torch_mlir.dynamo import make_simple_dynamo_backend
import torch._dynamo as dynamo
from torch.fx.experimental.proxy_tensor import make_fx
from torch_mlir_e2e_test.linalg_on_tensors_backends import refbackend
from shark.shark_inference import SharkInference

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionSafetyChecker,
)
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    CLIPFeatureExtractor,
    CLIPTextModel,
    CLIPTokenizer,
)


# Enter your HuggingFace Token
# Note: You can comment this prompt and just set your token instead of passing it through cli for every execution.


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


# `pretrained_model_name_or_path` which Stable Diffusion checkpoint you want to use
# Options: 1.) "stabilityai/stable-diffusion-2"
#          2.) "stabilityai/stable-diffusion-2-base"
#          3.) "CompVis/stable-diffusion-v1-4"
#          4.) "runwayml/stable-diffusion-v1-5"
pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"

# Add here the URLs to the images of the concept you are adding. 3-5 should be fine
urls = [
    "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
    "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
    ## You can add additional images here
]

# Downloading Images
import requests
import glob
from io import BytesIO


def download_image(url):
    try:
        response = requests.get(url)
    except:
        return None
    return Image.open(BytesIO(response.content)).convert("RGB")


images = list(filter(None, [download_image(url) for url in urls]))
save_path = "./my_concept"
if not os.path.exists(save_path):
    os.mkdir(save_path)
[image.save(f"{save_path}/{i}.jpeg") for i, image in enumerate(images)]

torch.manual_seed(args.seed)

if "*s" not in args.prompt:
    raise ValueError(
        f'The prompt should have a "*s" which will be replaced by a placeholder token.'
    )

prompt1, prompt2 = args.prompt.split("*s")
args.prompt = prompt1 + args.placeholder_token + prompt2

# `images_path` is a path to directory containing the training images.
from datasets import load_dataset
ds = load_dataset("lambdalabs/pokemon-blip-captions", split="train")

# Setup and check the images you have just added
images = [dp["image"].resize((512,512)) for dp in ds]
image_grid(images, 1, len(images))

########### Create Dataset ##########

# Setup the prompt templates for training
imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


# Setup the dataset
class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizers,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizers = tokenizers
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root)
        ]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]

        self.templates = (
            imagenet_style_templates_small
            if learnable_property == "style"
            else imagenet_templates_small
        )
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        text_input_ids_list = []
        uncond_input_ids_list = []

        # Tokenize prompt and negative prompt.
        for tokenizer in self.tokenizers:
            text_inputs = tokenizer(
                text,
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input = tokenizer(
                "food", # TODO(kh)
                padding="max_length",
                max_length=args.max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            uncond_input_ids = uncond_input.input_ids

            text_input_ids_list.extend([text_input_ids])
            uncond_input_ids_list.extend([uncond_input_ids])


        example["input_ids_1"] = text_input_ids_list[0]
        example["uncond_ids_1"] = uncond_input_ids_list[0]
        example["input_ids_2"] = text_input_ids_list[1]
        example["uncond_ids_2"] = uncond_input_ids_list[1]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[
                (h - crop) // 2 : (h + crop) // 2,
                (w - crop) // 2 : (w + crop) // 2,
            ]

        image = Image.fromarray(img)
        image = image.resize(
            (self.size, self.size), resample=self.interpolation
        )

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


########## Setting up the model ##########

# Load the tokenizer and add the placeholder token as a additional special token.
tokenizer_1 = CLIPTokenizer.from_pretrained(
    args.hf_model_name,
    subfolder="tokenizer",
    token=args.hf_auth_token,
)
tokenizer_2 = CLIPTokenizer.from_pretrained(
    args.hf_model_name,
    subfolder="tokenizer_2",
    token=args.hf_auth_token,
)

prompt_encoder = PromptEncoderModule(args.hf_model_name, args.precision, args.hf_auth_token)

# Add the placeholder token in tokenizer
#num_added_tokens_1 = tokenizer_1.add_tokens(args.placeholder_token)
#num_added_tokens_2 = tokenizer_2.add_tokens(args.placeholder_token)
#if num_added_tokens_1 == 0 or num_added_tokens_2 == 0:
#    raise ValueError(
#        f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
#        " `placeholder_token` that is not already in the tokenizer."
#    )

# Get token ids for our placeholder and initializer token.
# This code block will complain if initializer string is not a single token
# Convert the initializer_token, placeholder_token to ids
#token_ids_1 = tokenizer_1.encode(args.initializer_token, add_special_tokens=False)
#token_ids_2 = tokenizer_2.encode(args.initializer_token, add_special_tokens=False)
# Check if initializer_token is a single token or a sequence of tokens
#if len(token_ids_1) > 1 or len(token_ids_2) > 1:
#    raise ValueError("The initializer token must be a single token.")

#initializer_token_id_1 = token_ids_1[0]
#initializer_token_id_2 = token_ids_2[0]
#placeholder_token_id_1 = tokenizer_1.convert_tokens_to_ids(args.placeholder_token)
#placeholder_token_id_2 = tokenizer_2.convert_tokens_to_ids(args.placeholder_token)

# Load the Stable Diffusion model
# Load models and create wrapper for stable diffusion
# pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
# del pipeline

# We have added the placeholder_token in the tokenizer so we resize the token embeddings here
# this will a new embedding vector in the token embeddings for our placeholder_token
#prompt_encoder.text_encoder_model_1.resize_token_embeddings(len(tokenizer_1))
#prompt_encoder.text_encoder_model_2.resize_token_embeddings(len(tokenizer_2))

# Initialise the newly added placeholder token with the embeddings of the initializer token
#token_embeds_1 = prompt_encoder.text_encoder_model_1.get_input_embeddings().weight.data
#token_embeds_2 = prompt_encoder.text_encoder_model_2.get_input_embeddings().weight.data
#token_embeds_1[placeholder_token_id_1] = token_embeds_1[initializer_token_id_1]
#token_embeds_2[placeholder_token_id_2] = token_embeds_2[initializer_token_id_2]

# In Textual-Inversion we only train the newly added embedding vector
#  so lets freeze rest of the model parameters here


def freeze_params(params):
    for param in params:
        param.requires_grad = False


# Move vae and unet to device
# For the dynamo path default compilation device is `cpu`, since torch-mlir
# supports only that. Therefore, convert to device only for PyTorch path.
#if not args.use_torchdynamo:
#    vae.to(args.device)
#    unet.to(args.device)



vae_model = VaeModel(
    args.hf_model_name,
    args.hf_auth_token,
)
unet_model = UnetModel(
    args.hf_model_name,
    args.hf_auth_token,
)

# Keep vae in eval mode as we don't train it
vae_model.vae.eval()
# Keep unet in train mode to enable gradient checkpointing
unet_model.unet.train()
# Freeze vae and unet
freeze_params(vae_model.vae.parameters())
freeze_params(unet_model.unet.parameters())
freeze_params(prompt_encoder.text_encoder_model_2.parameters())
# Freeze all parameters except for the token embeddings in text encoder
params_to_freeze = itertools.chain(
    prompt_encoder.text_encoder_model_1.text_model.encoder.parameters(),
    prompt_encoder.text_encoder_model_1.text_model.final_layer_norm.parameters(),
    prompt_encoder.text_encoder_model_1.text_model.embeddings.position_embedding.parameters(),
#    prompt_encoder.text_encoder_model_2.text_model.encoder.parameters(),
#    prompt_encoder.text_encoder_model_2.text_model.final_layer_norm.parameters(),
#    prompt_encoder.text_encoder_model_2.text_model.embeddings.position_embedding.parameters(),
)
#freeze_params(params_to_freeze)

####### Creating our training data ########

# Let's create the Dataset and Dataloader
train_dataset = TextualInversionDataset(
    data_root=save_path,
    tokenizers=[tokenizer_1, tokenizer_2],
    size=vae_model.vae.sample_size,
    placeholder_token=args.placeholder_token,
    repeats=100,
    learnable_property=args.what_to_teach,  # Option selected above between object and style
    center_crop=False,
    set="train",
)


def create_dataloader(train_batch_size=1):
    return torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )


# Create noise_scheduler for training
noise_scheduler = DDPMScheduler.from_config(
    pretrained_model_name_or_path, subfolder="scheduler"
)

######## Training ###########

# Define hyperparameters for our training. If you are not happy with your results,
# you can tune the `learning_rate` and the `max_train_steps`

# Setting up all training args
hyperparameters = {
    "learning_rate": 5e-04,
    "scale_lr": True,
    "max_train_steps": args.training_steps,
    "save_steps": args.save_steps,
    "train_batch_size": args.train_batch_size,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "output_dir": "sd-concept-output",
}
# creating output directory
cwd = os.getcwd()
out_dir = os.path.join(cwd, hyperparameters["output_dir"])
while not os.path.exists(str(out_dir)):
    try:
        os.mkdir(out_dir)
    except OSError as error:
        print("Output directory not created")

###### Torch-MLIR Compilation ######


def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
    removed_indexes = []
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert (
                len(node.args) == 1
            ), "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, (list, tuple)):
                node_arg = list(node_arg)
                node_args_len = len(node_arg)
                for i in range(node_args_len):
                    curr_index = node_args_len - (i + 1)
                    if node_arg[curr_index] is None:
                        removed_indexes.append(curr_index)
                        node_arg.pop(curr_index)
                node.args = (tuple(node_arg),)
                break

    if len(removed_indexes) > 0:
        fx_g.graph.lint()
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
    removed_indexes.sort()
    return removed_indexes


def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    """
    Replace tuple with tuple element in functions that return one-element tuples.
    Returns true if an unwrapping took place, and false otherwise.
    """
    unwrapped_tuple = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert (
                len(node.args) == 1
            ), "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    unwrapped_tuple = True
                    break

    if unwrapped_tuple:
        fx_g.graph.lint()
        fx_g.recompile()
    return unwrapped_tuple


def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert (
                len(node.args) == 1
            ), "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                return len(node_arg) == 0
    return False


def transform_fx(fx_g):
    for node in fx_g.graph.nodes:
        if node.op == "call_function":
            if node.target in [
                torch.ops.aten.empty,
            ]:
                # aten.empty should be filled with zeros.
                if node.target in [torch.ops.aten.empty]:
                    with fx_g.graph.inserting_after(node):
                        new_node = fx_g.graph.call_function(
                            torch.ops.aten.zero_,
                            args=(node,),
                        )
                        node.append(new_node)
                        node.replace_all_uses_with(new_node)
                        new_node.args = (node,)

    fx_g.graph.lint()


@make_simple_dynamo_backend
def refbackend_torchdynamo_backend(
    fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    # handling usage of empty tensor without initializing
    transform_fx(fx_graph)
    fx_graph.recompile()
    if _returns_nothing(fx_graph):
        return fx_graph
    removed_none_indexes = _remove_nones(fx_graph)
    was_unwrapped = _unwrap_single_tuple_return(fx_graph)

    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors"
    )

    bytecode_stream = BytesIO()
    mlir_module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    shark_module = SharkInference(
        mlir_module=bytecode, device=args.device, mlir_dialect="tm_tensor"
    )
    shark_module.compile()

    def compiled_callable(*inputs):
        inputs = [x.numpy() for x in inputs]
        result = shark_module("forward", inputs)
        if was_unwrapped:
            result = [
                result,
            ]
        if not isinstance(result, list):
            result = torch.from_numpy(result)
        else:
            result = tuple(torch.from_numpy(x) for x in result)
            result = list(result)
            for removed_index in removed_none_indexes:
                result.insert(removed_index, None)
            result = tuple(result)
        return result

    return compiled_callable


def predictions(torch_func, jit_func, batchA, batchB):
    res = jit_func(batchA.numpy(), batchB.numpy())
    if res is not None:
        prediction = res
    else:
        prediction = None
    return prediction


logger = logging.getLogger(__name__)


# def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
def save_progress(text_encoder, placeholder_token_id, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        # accelerator.unwrap_model(text_encoder)
        text_encoder.get_input_embeddings().weight[placeholder_token_id]
    )
    learned_embeds_dict = {
        args.placeholder_token: learned_embeds.detach().cpu()
    }
    torch.save(learned_embeds_dict, save_path)


train_batch_size = hyperparameters["train_batch_size"]
gradient_accumulation_steps = hyperparameters["gradient_accumulation_steps"]
learning_rate = hyperparameters["learning_rate"]
if hyperparameters["scale_lr"]:
    learning_rate = (
        learning_rate
        * gradient_accumulation_steps
        * train_batch_size
        # * accelerator.num_processes
    )

# Initialize the optimizer
#optimizer = torch.optim.AdamW(
#    list(prompt_encoder.text_encoder_model_1.get_input_embeddings().parameters()) +
#            list(prompt_encoder.text_encoder_model_2.get_input_embeddings().parameters()),  # only optimize the embeddings
#    lr=learning_rate,
#)
optimizer = torch.optim.AdamW(
    list(prompt_encoder.text_encoder_model_1.get_input_embeddings().parameters()),
    lr=learning_rate,
)

from turbine_models.custom_models.sd_inference import utils
def run_torch_diffusers_loop(
    sample,
    prompt_embeds,
    text_embeds,
    args,
):
    scheduler = utils.get_schedulers(args.hf_model_name)[args.scheduler_id]

    scheduler.set_timesteps(args.num_inference_steps)
    scheduler.is_scale_input_called = True
    sample = sample * scheduler.init_noise_sigma

    height = sample.shape[-2] * 8
    width = sample.shape[-1] * 8
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)

    add_time_ids = list(original_size + crops_coords_top_left + target_size)
    add_time_ids = torch.tensor([add_time_ids, add_time_ids], dtype=torch.float32)
    add_time_ids = add_time_ids.repeat(args.batch_size * 1, 1)
    sample = sample.to(torch.float32)
    prompt_embeds = prompt_embeds.to(torch.float32)
    text_embeds = text_embeds.to(torch.float32)

    for i in range(args.num_inference_steps):
        timestep = scheduler.timesteps[i]

        latent_model_input = scheduler.scale_model_input(sample, timestep)
        print("sizes")
        print(latent_model_input.shape)
        print(timestep.shape)
        print(prompt_embeds.shape)
        print(text_embeds.shape)
        print(add_time_ids.shape)
        print(args.guidance_scale)
        noise_pred = unet_model.forward(
            latent_model_input,
            timestep,
            prompt_embeds,
            text_embeds,
            add_time_ids,
            args.guidance_scale,
        )
        sample = scheduler.step(
            noise_pred,
            timestep,
            sample,
            return_dict=False,
        )[0]
    return sample

# Training function
def train_func(pixel_values, samples, input_ids_1, input_ids_2, uncond_ids_1, uncond_ids_2):
    # Convert images to latent space
    print(pixel_values.shape)
    latents = vae_model.encode_inp(pixel_values)

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(
        0,
        noise_scheduler.num_train_timesteps,
        (bsz,),
        device=latents.device,
    ).long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Get the text embedding for conditioning
    #encoder_hidden_states = text_encoder(batch_input_ids)[0]
    print(input_ids_1.shape)
    print(input_ids_2.shape)
    print(uncond_ids_1.shape)
    print(uncond_ids_2.shape)
    prompt_embeds = [i for i in range(input_ids_1.shape[0])]
    text_embeds = [i for i in range(input_ids_1.shape[0])]
    for i in range(input_ids_1.shape[0]):
        prompt_embeds, text_embeds = prompt_encoder(input_ids_1[i], input_ids_2[i], uncond_ids_1[i], uncond_ids_2[i])
    #prompt_embeds, text_embeds = prompt_encoder(input_ids_1, input_ids_2, uncond_ids_1, uncond_ids_2)

    # Predict the noise residual
    #noise_pred = unet_model(
    #    noisy_latents,
    #    timesteps,
    #    encoder_hidden_states,
    #)
    noise_preds = run_torch_diffusers_loop(samples, prompt_embeds, text_embeds, args) # TODO(kh): batch

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )

    print(target.shape)
    print(noise_preds.shape)
    loss = (
        F.mse_loss(noise_preds, target, reduction="none").mean([1, 2, 3]).mean()
    )
    loss.backward()

    # Zero out the gradients for all token embeddings except the newly added
    # embeddings for the concept, as we only want to optimize the concept embeddings
    #grads = text_encoder.get_input_embeddings().weight.grad
    # Get the index for tokens that we want to zero the grads for
    #index_grads_to_zero = torch.arange(len(tokenizer_1)) != placeholder_token_id # TODO(kh)
    #grads.data[index_grads_to_zero, :] = grads.data[
    #    index_grads_to_zero, :
    #].fill_(0)

    optimizer.step()
    optimizer.zero_grad()

    return loss

print("DEBUG0")
class TrainModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return train_func(x, y)
print("DEBUG1")

def training_function():
    max_train_steps = hyperparameters["max_train_steps"]
    output_dir = hyperparameters["output_dir"]
    gradient_checkpointing = hyperparameters["gradient_checkpointing"]

    train_dataloader = create_dataloader(train_batch_size)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        train_batch_size
        * gradient_accumulation_steps
        # train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )
    print("DEBUG2")

    #logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataset)}")
    #logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    #logger.info(
    #    f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    #)
    #logger.info(
    #    f"  Gradient Accumulation steps = {gradient_accumulation_steps}"
    #)
    #logger.info(f"  Total optimization steps = {max_train_steps}")
    #Only show the progress bar once on each machine.
    progress_bar = tqdm(
        # range(max_train_steps), disable=not accelerator.is_local_main_process
        range(max_train_steps)
    )
    progress_bar.set_description("Steps")
    global_step = 0
    print("DEBUG3")

    #params_ = [i for i in prompt_encoder.text_encoder_model_1.get_input_embeddings().parameters()]
    #params_ = list(prompt_encoder.text_encoder_model_1.get_input_embeddings().parameters()) + list(prompt_encoder.text_encoder_model_2.get_input_embeddings().parameters())  # only optimize the embeddings
    #print("DEBUG4")
    if args.use_torchdynamo:
        print("******** TRAINING STARTED - TORCHYDNAMO PATH ********")
    else:
        print("******** TRAINING STARTED - PYTORCH PATH ********")
    #print("Initial weights:")
    #print(params_, params_[0].shape)

    for epoch in range(num_train_epochs):
        prompt_encoder.train()
        for step, batch in enumerate(train_dataloader):
            if args.use_torchdynamo:
                train_model = TrainModel()
                print("DEBUG T0")
                print(args.batch_size)
                generator = torch.manual_seed(args.seed + step)
                samples = torch.randn(
                    (
                        args.batch_size,
                        4,
                        args.height // 8,
                        args.width // 8,
                    ),
                    generator=generator,
                    dtype=torch.float32, #todo
                )
                print("DEBUG T00")
                train_func(batch["pixel_values"], samples, batch["input_ids_1"], batch["input_ids_2"], batch["uncond_ids_1"], batch["uncond_ids_2"])
                print("DEBUG T01")
                class CompiledTraining(CompiledModule):
                    params = export_parameters(train_model)

                    def main(
                        self,
                        pixel_values=AbstractTensor(*batch["pixel_values"].shape, dtype=torch.float32),
                        samples=AbstractTensor(*samples.shape, dtype=torch.float32),
                        input_ids_1=AbstractTensor(*batch["input_ids_1"].shape, dtype=torch.float32),
                        input_ids_2=AbstractTensor(*batch["input_ids_2"].shape, dtype=torch.float32),
                        uncond_ids_1=AbstractTensor(*batch["uncond_ids_1"].shape, dtype=torch.float32),
                        uncond_ids_2=AbstractTensor(*batch["uncond_ids_2"].shape, dtype=torch.float32)
                    ):
                        return jittable(train_func, decompose_ops=DEFAULT_DECOMPOSITIONS)(
                            pixel_values, samples, input_ids_1, input_ids_2, uncond_ids_1, uncond_ids_2
                        )
                print("DEBUG T1")
                import_to = "INPUT" if args.compile_to == "linalg" else "IMPORT"
                inst = CompiledTraining(context=Context(), import_to=import_to)
                print("DEBUG T2")

                module_str = str(CompiledModule.get_mlir_module(inst))

                if compile_to != "vmfb":
                    return module_str
                else:
                    utils.compile_to_vmfb(
                        module_str,
                        device,
                        target_triple,
                        ireec_flags,
                        safe_name,
                        return_path=False,
                        attn_spec=attn_spec,
                    )
                print("DEBUG T3")
                dynamo_callable = dynamo.optimize(
                    refbackend_torchdynamo_backend
                )(train_func)
                lam_func = lambda x, y: dynamo_callable(
                    torch.from_numpy(x), torch.from_numpy(y)
                )
                loss = predictions(
                    train_func,
                    lam_func,
                    batch["pixel_values"],
                    batch["input_ids"],
                    # params[0].detach(),
                )
            else:
                loss = train_func(batch["pixel_values"], batch["input_ids"])
            print(loss)

            # Checks if the accelerator has performed an optimization step behind the scenes
            progress_bar.update(1)
            global_step += 1
            if global_step % hyperparameters["save_steps"] == 0:
                save_path = os.path.join(
                    output_dir,
                    f"learned_embeds-step-{global_step}.bin",
                )
                save_progress(
                    text_encoder,
                    placeholder_token_id,
                    save_path,
                )

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    #params__ = [i for i in text_encoder.get_input_embeddings().parameters()]
    #print("******** TRAINING PROCESS FINISHED ********")
    ##print("Updated weights:")
    #print(params__, params__[0].shape)
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        # text_encoder=accelerator.unwrap_model(text_encoder),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        unet=unet,
    )
    pipeline.save_pretrained(output_dir)
    # Also save the newly trained embeddings
    save_path = os.path.join(output_dir, f"learned_embeds.bin")
    save_progress(text_encoder, placeholder_token_id, save_path)


training_function()

for param in itertools.chain(unet.parameters(), text_encoder.parameters()):
    if param.grad is not None:
        del param.grad  # free some memory
    torch.cuda.empty_cache()

# Set up the pipeline
from diffusers import DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    hyperparameters["output_dir"],
    scheduler=DPMSolverMultistepScheduler.from_pretrained(
        hyperparameters["output_dir"], subfolder="scheduler"
    ),
)
if not args.use_torchdynamo:
    pipe.to(args.device)

# Run the Stable Diffusion pipeline
# Don't forget to use the placeholder token in your prompt

all_images = []
for _ in range(args.num_inference_samples):
    images = pipe(
        [args.prompt],
        num_inference_steps=args.inference_steps,
        guidance_scale=7.5,
    ).images
    all_images.extend(images)

output_path = os.path.abspath(os.path.join(os.getcwd(), args.output_dir))
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)

[
    image.save(f"{args.output_dir}/{i}.jpeg")
    for i, image in enumerate(all_images)
]
