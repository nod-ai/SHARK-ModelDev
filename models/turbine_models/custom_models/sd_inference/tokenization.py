from typing import List, Optional, Union
from iree import runtime as ireert
import re
import torch
import numpy as np
import warnings


# The following is copied from Diffusers' "encode_prompt" function in the StableDiffusion pipeline.
# It has been lightly augmented to work with the SHARK-Turbine pipeline.
def encode_prompt(
    pipe,
    prompt,
    negative_prompt=None,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    lora_scale: Optional[float] = None,
    clip_skip: Optional[int] = None,
):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            prompt to be encoded
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.Tensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        lora_scale (`float`, *optional*):
            A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        clip_skip (`int`, *optional*):
            Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
            the output of the pre-final layer will be used for computing the prompt embeddings.
    """
    # set lora scale so that monkey patched LoRA
    # function of text encoder can correctly access it
    # if lora_scale is not None and pipe.use_lora:
    #     pipe._lora_scale = lora_scale

    #     # dynamically adjust the LoRA scale
    #     if not USE_PEFT_BACKEND:
    #         adjust_lora_scale_text_encoder(pipe.text_encoder, lora_scale)
    #     else:
    #         scale_lora_layers(pipe.text_encoder, lora_scale)

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        # textual inversion: process multi-vector tokens if necessary
        # if pipe.use_textual_inversion:
        #     prompt = pipe.maybe_convert_prompt(prompt, pipe.tokenizer)

        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = pipe.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = pipe.tokenizer.batch_decode(
                untruncated_ids[:, pipe.model_max_length - 1 : -1]
            )
            warnings.warn(
                "The following text was removed due to truncation: " + removed_text
            )
        if pipe.text_encoder.metadata.get("use_attention_mask"):
            attention_mask = text_inputs.attention_mask
            prompt_embeds = pipe.text_encoder(
                "encode_tokens_attn_mask", [text_input_ids, attention_mask]
            )
        else:
            attention_mask = None
            prompt_embeds = pipe.text_encoder("encode_tokens", [text_input_ids])
        prompt_embeds = prompt_embeds[0]
    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        # textual inversion: process multi-vector tokens if necessary
        # if pipe.use_textual_inversion:
        #     uncond_tokens = pipe.maybe_convert_prompt(uncond_tokens, pipe.tokenizer)

        max_length = prompt_embeds.shape[1]
        uncond_input = pipe.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if pipe.text_encoder.metadata.get("use_attention_mask"):
            attention_mask = uncond_input.attention_mask
            negative_prompt_embeds = pipe.text_encoder(
                "encode_tokens_attn_mask",
                [
                    uncond_input.input_ids,
                    attention_mask,
                ],
            )
        else:
            attention_mask = None
            negative_prompt_embeds = pipe.text_encoder(
                "encode_tokens",
                [
                    uncond_input.input_ids,
                ],
            )

        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.repeat(
            1, num_images_per_prompt, 1
        )
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

    # if pipe.use_lora:
    # Retrieve the original scale by scaling back the LoRA layers
    # unimplemented
    # unscale_lora_layers(pipe.text_encoder, lora_scale)

    return prompt_embeds, negative_prompt_embeds
