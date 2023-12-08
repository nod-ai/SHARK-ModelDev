#!/usr/bin/env python3
from enum import IntEnum, auto


class MODEL_ARCH(IntEnum):
    LLAMA: int = auto()
    FALCON: int = auto()
    BAICHUAN: int = auto()
    GPT2: int = auto()
    GPTJ: int = auto()
    GPTNEOX: int = auto()
    MPT: int = auto()
    STARCODER: int = auto()
    PERSIMMON: int = auto()
    REFACT: int = auto()
    BERT: int = auto()
    BLOOM: int = auto()


class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD: int = auto()
    TOKEN_EMBD_NORM: int = auto()
    TOKEN_TYPES: int = auto()
    POS_EMBD: int = auto()
    OUTPUT: int = auto()
    OUTPUT_NORM: int = auto()
    ROPE_FREQS: int = auto()
    ATTN_Q: int = auto()
    ATTN_K: int = auto()
    ATTN_V: int = auto()
    ATTN_QKV: int = auto()
    ATTN_OUT: int = auto()
    ATTN_NORM: int = auto()
    ATTN_NORM_2: int = auto()
    ATTN_ROT_EMBD: int = auto()
    FFN_GATE: int = auto()
    FFN_DOWN: int = auto()
    FFN_UP: int = auto()
    FFN_NORM: int = auto()
    ATTN_Q_NORM: int = auto()
    ATTN_K_NORM: int = auto()


MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.LLAMA: "llama-hf",
    MODEL_ARCH.FALCON: "falcon",
    MODEL_ARCH.BAICHUAN: "baichuan",
    MODEL_ARCH.GPT2: "gpt2",
    MODEL_ARCH.GPTJ: "gptj",
    MODEL_ARCH.GPTNEOX: "gptneox",
    MODEL_ARCH.MPT: "mpt",
    MODEL_ARCH.STARCODER: "starcoder",
    MODEL_ARCH.PERSIMMON: "persimmon",
    MODEL_ARCH.REFACT: "refact",
    MODEL_ARCH.BERT: "bert",
    MODEL_ARCH.BLOOM: "bloom",
}

TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD: "token_embd.weight",
    MODEL_TENSOR.TOKEN_EMBD_NORM: "token_embd_norm.weight",
    MODEL_TENSOR.TOKEN_TYPES: "token_types.weight",
    MODEL_TENSOR.POS_EMBD: "position_embd.weight",
    MODEL_TENSOR.OUTPUT_NORM: "output_norm.weight",
    MODEL_TENSOR.OUTPUT: "output.weight",
    MODEL_TENSOR.ROPE_FREQS: "rope_freqs.weight",
    MODEL_TENSOR.ATTN_NORM: "blk.{bid}.attn_norm.weight",
    MODEL_TENSOR.ATTN_NORM_2: "blk.{bid}.attn_norm_2.weight",
    MODEL_TENSOR.ATTN_QKV: "blk.{bid}.attn_qkv.weight",
    MODEL_TENSOR.ATTN_Q: "blk.{bid}.attn_q.weight",
    MODEL_TENSOR.ATTN_K: "blk.{bid}.attn_k.weight",
    MODEL_TENSOR.ATTN_V: "blk.{bid}.attn_v.weight",
    MODEL_TENSOR.ATTN_OUT: "blk.{bid}.attn_output.weight",
    MODEL_TENSOR.ATTN_ROT_EMBD: "blk.{bid}.attn_rot_embd.weight",
    MODEL_TENSOR.ATTN_Q_NORM: "blk.{bid}.attn_q_norm.weight",
    MODEL_TENSOR.ATTN_K_NORM: "blk.{bid}.attn_k_norm.weight",
    MODEL_TENSOR.FFN_NORM: "blk.{bid}.ffn_norm.weight",
    MODEL_TENSOR.FFN_GATE: "blk.{bid}.ffn_gate.weight",
    MODEL_TENSOR.FFN_DOWN: "blk.{bid}.ffn_down.weight",
    MODEL_TENSOR.FFN_UP: "blk.{bid}.ffn_up.weight",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPTNEOX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.FALCON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STARCODER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MPT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPTJ: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PERSIMMON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.REFACT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BLOOM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPT2: [
        # TODO
    ],
    # TODO
}


# TODO non llama-hf mapping needs to be verified and likely need small changes
class TensorNameMap:
    mappings_cfg: dict[MODEL_TENSOR, dict[str, str]] = {
        # Token embeddings
        MODEL_TENSOR.TOKEN_EMBD: {
            "gpt2": "transformer.wte",
            "falcon": "transformer.word_embeddings",
            "bloom": "word_embeddings",
            "llama-hf": "params.model.embed_tokens.weight",
            "llama-pth": "tok_embeddings",
            "bert": "embeddings.word_embeddings",
            "persimmon": "language_model.embedding.word_embeddings",
        },
        # Token type embeddings
        MODEL_TENSOR.TOKEN_TYPES: {
            "bert": "embeddings.token_type_embeddings",
        },
        # Normalization of token embeddings
        MODEL_TENSOR.TOKEN_EMBD_NORM: {
            "bloom": "word_embeddings_layernorm",
        },
        # Position embeddings
        MODEL_TENSOR.POS_EMBD: {
            "gpt2": "transformer.wpe",
            "bert": "embeddings.position_embeddings",
        },
        # Output
        MODEL_TENSOR.OUTPUT: {
            "gptneoz": "embed_out",
            "gpt2": "params.lm_head.weight",
            "mpt": "params.lm_head.weight",
            "falcon": "params.lm_head.weight",
            "llama-hf": "params.lm_head.weight",
            "baichuan": "params.lm_head.weight",
            "llama-pth": "output",
            "persimmon": "word_embeddings_for_head",
        },
        # Output norm
        MODEL_TENSOR.OUTPUT_NORM: {
            "gptneoz": "gpt_neox.final_layer_norm",
            "gpt2": "transformer.ln_f",
            "gpt-j": "transformer.ln_f",
            "falcon": "transformer.ln_f",
            "llama-hf": "params.model.norm.weight",
            "baichuan": "params.model.norm.weight",
            "llama-pth": "norm",
            "bert": "embeddings.LayerNorm",
            "mpt": "transformer.norm_f",
            "refact": "ln_f",
            "bloom": "ln_f",
            "persimmon": "language_model.encoder.final_layernorm",
        },
        # Rope frequencies
        MODEL_TENSOR.ROPE_FREQS: {
            "llama-pth": "params.model.rope.freqs",
        },
    }

    block_mappings_cfg: dict[MODEL_TENSOR, dict[str, str]] = {
        # Attention norm
        MODEL_TENSOR.ATTN_NORM: {
            "gptneox": "gpt_neox.layers.{bid}.input_layernorm",
            "gpt2": "transformer.h.{bid}.ln_1",
            "refact": "transformer.h.{bid}.ln_1",
            "gpt2-j": "transformer.h.{bid}.ln_1",
            "mpt": "transformer.blocks.{bid}.norm_1",
            "falcon7b": "transformer.h.{bid}.input_layernorm",
            "bloom": "h.{bid}.input_layernorm",
            "falcon40b": "transformer.h.{bid}.ln_mlp",
            "llama-hf": "params.model.layers.{bid}.input_layernorm.weight",
            "llama-pth": "layers.{bid}.attention_norm",
            "bert": "encoder.layer.{bid}.attention.output.LayerNorm",
            "persimmon": "language_model.encoder.layers.{bid}.input_layernorm",
        },
        # Attention norm 2
        MODEL_TENSOR.ATTN_NORM_2: {
            "falcon40b": "transformer.h.{bid}.ln_attn",
        },
        # Attention query-key-value
        MODEL_TENSOR.ATTN_QKV: {
            "gptneox": "gpt_neox.layers.{bid}.attention.query_key_value",
            "mpt": "transformer.h.{bid}.attn.c_attn",
            "falcon": "transformer.blocks.{bid}.attn.Wqkv",
            "falcon": "transformer.h.{bid}.self_attention.query_key_value",
            "bloom": "h.{bid}.self_attention.query_key_value",
            "persimmon": "language_model.encoder.layers.{bid}.self_attention.query_key_value",
        },
        # Attention query
        MODEL_TENSOR.ATTN_Q: {
            "llama-hf": "params.model.layers.{bid}.self_attn.q_proj.weight",
            "llama-pth": "layers.{bid}.attention.wq",
            "bert": "encoder.layer.{bid}.attention.self.query",
            "gpt-j": "transformer.h.{bid}.attn.q_proj",
        },
        # Attention key
        MODEL_TENSOR.ATTN_K: {
            "llama-hf": "params.model.layers.{bid}.self_attn.k_proj.weight",
            "llama-pth": "layers.{bid}.attention.wk",
            "bert": "encoder.layer.{bid}.attention.self.key",
            "gpt-j": "transformer.h.{bid}.attn.k_proj",
        },
        # Attention value
        MODEL_TENSOR.ATTN_V: {
            "llama-hf": "params.model.layers.{bid}.self_attn.v_proj.weight",
            "llama-pth": "layers.{bid}.attention.wv",
            "bert": "encoder.layer.{bid}.attention.self.value",
            "gpt-j": "transformer.h.{bid}.attn.v_proj",
        },
        # Attention output
        MODEL_TENSOR.ATTN_OUT: {
            "gptneox": "gpt_neox.layers.{bid}.attention.dense",
            "gpt2": "transformer.h.{bid}.attn.c_proj",
            "refact": "transformer.h.{bind}.attn.c_proj",
            "mpt": "transformer.blocks.{bid}.attn.out_proj",
            "falcon": "transformer.h.{bid}.self_attention.dense",
            "bloom": "h.{bid}.self_attention.dense",
            "llama-hf": "params.model.layers.{bid}.self_attn.o_proj.weight",
            "llama-pth": "layers.{bid}.attention.wo",
            "bert": "encoder.layer.{bid}.attention.output.dense",
            "gpt-j": "transformer.h.{bid}.attn.out_proj",
            "persimmon": "language_model.encoder.layers.{bid}.self_attention.dense",
        },
        # Rotary embeddings
        MODEL_TENSOR.ATTN_ROT_EMBD: {
            "llama-hf": "params.model.layers.{bid}.self_attn.rotary_emb.inv_freq.weight",
            "llama-pth": "layers.{bid}.attention.inner_attention.rope.freqs",
        },
        # Feed-forward norm
        MODEL_TENSOR.FFN_NORM: {
            "gptneox": "gpt_neox.layers.{bid}.post_attention_layernorm",
            "gpt2": "transformer.h.{bid}.ln_2",
            "refact": "transformer.h.{bid}.ln_2",
            "blom": "h.{bid}.post_attention_layernorm",
            "mpt": "transformer.blocks.{bid}.norm_2",
            "llama-hf": "params.model.layers.{bid}.post_attention_layernorm.weight",
            "llama-pth": "layers.{bid}.ffn_norm",
            "bert": "encoder.layer.{bid}.output.LayerNorm",
            "persimmon": "language_model.encoder.layers.{bid}.post_attention_layernorm",
        },
        # Feed-forward up
        MODEL_TENSOR.FFN_UP: {
            "gptneox": "gpt_neox.layers.{bid}.mlp.dense_h_to_4h",
            "gpt2": "transformer.h.{bid}.mlp.c_fc",
            "mpt": "transformer.blocks.{bid}.ffn.up_proj",
            "falcon": "transformer.h.{bid}.mlp.dense_h_to_4h",
            "bloom": "h.{bid}.mlp.dense_h_to_4h",
            "llama-hf": "params.model.layers.{bid}.mlp.up_proj.weight",
            "refact": "params.model.layer.{bid}.mlp.up_proj.weight",
            "llama-pth": "layers.{bid}.feed_forward.w3",
            "bert": "encoder.layer.{bid}.intermediate.dense",
            "gpt-j": "transformer.h.{bid}.mlp.fc_in",
            "persimmon": "language_model.encoder.layers.{bid}.mlp.dense_h_to_4h",
        },
        # Feed-forward gate
        MODEL_TENSOR.FFN_GATE: {
            "llama-hf": "params.model.layers.{bid}.mlp.gate_proj.weight",
            "refact": "params.model.layers.{bid}.mlp.gate_proj.weight",
            "llama-pth": "layers.{bid}.feed_forward.w1",
        },
        # Feed-forward down
        MODEL_TENSOR.FFN_DOWN: {
            "gptneox": "gpt_neox.layers.{bid}.mlp.dense_4h_to_h",
            "gpt2": "transformer.h.{bid}.mlp.c_proj",
            "refact": "transformer.h.{bid}.mlp.c_proj",
            "mpt": "transformer.blocks.{bid}.ffn.down_proj",
            "falcon": "transformer.h.{bid}.mlp.dense_4h_to_h",
            "bloom": "h.{bid}.mlp.dense_4h_to_h",
            "llama-hf": "params.model.layers.{bid}.mlp.down_proj.weight",
            "llama-pth": "layers.{bid}.feed_forward.w2",
            "bert": "params.encoder.layer.{bid}.output.dense",
            "gpt-j": "params.transformer.h.{bid}.mlp.fc_out",
            "persimmon": "params.language_model.encoder.layers.{bid}.mlp.dense_4h_to_h",
        },
    }

    def __init__(self, arch: MODEL_ARCH, n_blocks: int):
        self.mapping = {}
        for tensor, tensor_dict in self.mappings_cfg.items():
            if tensor not in MODEL_TENSORS[arch]:
                continue
            gguf_tensor_name = TENSOR_NAMES[tensor]
            if MODEL_ARCH_NAMES[arch] in tensor_dict:
                self.mapping[tensor_dict[MODEL_ARCH_NAMES[arch]]] = gguf_tensor_name
        for bid in range(n_blocks):
            for tensor, tensor_dict in self.block_mappings_cfg.items():
                if tensor not in MODEL_TENSORS[arch]:
                    continue
                gguf_tensor_name = TENSOR_NAMES[tensor].format(bid=bid)
                if MODEL_ARCH_NAMES[arch] in tensor_dict:
                    self.mapping[
                        tensor_dict[MODEL_ARCH_NAMES[arch]].format(bid=bid)
                    ] = gguf_tensor_name
