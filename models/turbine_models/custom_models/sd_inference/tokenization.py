from typing import List, Optional, Union
from iree import runtime as ireert
import re
import torch
import numpy as np

re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs:
        text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(tokenizer, prompt: List[str], max_length: int):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = tokenizer(word).input_ids[1:-1]
            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        print(
            "Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples"
        )
    return tokens, weights


def pad_tokens_and_weights(
    tokens,
    weights,
    max_length,
    bos,
    eos,
    no_boseos_middle=True,
    chunk_length=77,
):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = (
        max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    )
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] * (max_length - 1 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][
                        j
                        * (chunk_length - 2) : min(
                            len(weights[i]), (j + 1) * (chunk_length - 2)
                        )
                    ]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe,
    text_input,
    chunk_length: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[
                :, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2
            ].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            text_input_chunk[:, -1] = text_input[0, -1]

            text_input_chunk = ireert.asdevicearray(
                pipe.runners["clip"].config.device, text_input_chunk, "int64"
            )
            text_embedding = (
                pipe.runners["clip"]
                .ctx.modules.compiled_clip["main"](text_input_chunk)
            )[0].to_host()
            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        # SHARK: Convert the result to tensor
        # text_embeddings = torch.concat(text_embeddings, axis=1)
        text_embeddings_np = np.concatenate(np.array(text_embeddings))
        text_embeddings = torch.from_numpy(text_embeddings_np)
    else:
        text_input = ireert.asdevicearray(
            pipe.runners["clip"].config.device, text_input, "int64"
        )
        text_embeddings = (
            pipe.runners["clip"]
            .ctx.modules.compiled_clip["main"](text_input)
        )[0].to_host()
        text_embeddings = torch.from_numpy(text_embeddings)
    return text_embeddings


# This function deals with NoneType values occuring in tokens after padding
# It switches out None with 49407 as truncating None values causes matrix dimension errors,
def filter_nonetype_tokens(tokens: List[List]):
    return [[49407 if token is None else token for token in tokens[0]]]


def get_tokenized_inputs(
    pipe,
    tokenizer,
    prompt,
    uncond_prompt,
    max_length,
    max_embeddings_multiples: Optional[int] = 8,
    no_boseos_middle: Optional[bool] = True,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
):
    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(
            tokenizer, prompt, max_length - 2
        )
        if uncond_prompt is not None:
            uncond_tokens, uncond_weights = get_prompts_with_weights(
                tokenizer, uncond_prompt, max_length - 2
            )
    else:
        prompt_tokens = [
            token[1:-1]
            for token in tokenizer(
                prompt, max_length=max_length, truncation=True
            ).input_ids
        ]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1]
                for token in tokenizer(
                    uncond_prompt, max_length=max_length, truncation=True
                ).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))
    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)

    max_length = (pipe.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.model_max_length,
    )

    # FIXME: This is a hacky fix caused by tokenizer padding with None values
    prompt_tokens = filter_nonetype_tokens(prompt_tokens)

    # prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device="cpu")
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.model_max_length,
        )

        # FIXME: This is a hacky fix caused by tokenizer padding with None values
        uncond_tokens = filter_nonetype_tokens(uncond_tokens)

        # uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device="cpu")
    if uncond_prompt is not None:
        return prompt_tokens, prompt_weights, uncond_tokens, uncond_weights
    else:
        return prompt_tokens, prompt_weights, None, None


def get_weighted_text_embeddings(
    pipe,
    prompt: List[str],
    uncond_prompt: List[str] = None,
    max_embeddings_multiples: Optional[int] = 8,
    no_boseos_middle: Optional[bool] = True,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
):
    max_length = (pipe.model_max_length - 2) * max_embeddings_multiples + 2
    for tokenizer in pipe.tokenizers:
        (
            prompt_tokens,
            prompt_weights,
            uncond_tokens,
            uncond_weights,
        ) = get_tokenized_inputs(
            pipe,
            tokenizer,
            prompt,
            uncond_prompt,
            max_length,
            max_embeddings_multiples,
            no_boseos_middle,
            skip_parsing,
            skip_weighting,
        )

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.model_max_length,
        no_boseos_middle=no_boseos_middle,
    )
    # prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
    prompt_weights = torch.tensor(prompt_weights, dtype=torch.float, device="cpu")
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.model_max_length,
            no_boseos_middle=no_boseos_middle,
        )
        # uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)
        uncond_weights = torch.tensor(uncond_weights, dtype=torch.float, device="cpu")

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = (
            text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        )
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = (
            text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        )
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = (
                uncond_embeddings.float()
                .mean(axis=[-2, -1])
                .to(uncond_embeddings.dtype)
            )
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = (
                uncond_embeddings.float()
                .mean(axis=[-2, -1])
                .to(uncond_embeddings.dtype)
            )
            uncond_embeddings *= (
                (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
            )

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings
    return text_embeddings, None
