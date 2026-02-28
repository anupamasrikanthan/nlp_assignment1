from __future__ import annotations
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

# Local module imports
from cs336_basics.model import (
    Linear, RMSNorm, Embedding, scaled_dot_product_attention,
    SwiGLU, CausalMultiHeadSelfAttention, RotaryPositionalEmbedding,
    TransformerBlock, TransformerLM
)
from cs336_basics.nn_utils import (
    softmax, cross_entropy, clip_gradient_norm, 
    save_checkpoint, load_checkpoint, silu
)
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.bpe import BPETokenizer, BPETokenizerParams, train_bpe


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    layer = Linear(d_in, d_out)
    layer.weight.data = weights
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    layer = Embedding(vocab_size, d_model)
    layer.weight.data = weights 
    return layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    ffn = SwiGLU(d_model, d_ff)
    ffn.w1.weight.data = w1_weight
    ffn.w2.weight.data = w2_weight
    ffn.w3.weight.data = w3_weight
    return ffn(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    attn = CausalMultiHeadSelfAttention(d_model, num_heads)
    attn.q_proj.weight.data = q_proj_weight
    attn.k_proj.weight.data = k_proj_weight
    attn.v_proj.weight.data = v_proj_weight
    attn.output_proj.weight.data = o_proj_weight
    return attn(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    attn = CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        rope_theta=theta,
    )
    attn.q_proj.weight.data = q_proj_weight
    attn.k_proj.weight.data = k_proj_weight
    attn.v_proj.weight.data = v_proj_weight
    attn.output_proj.weight.data = o_proj_weight
    
    return attn(in_features, pos_ids=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope = RotaryPositionalEmbedding(d_k, theta=theta, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    block = TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    block.load_state_dict(weights, strict=False)
    
    # Generate the missing pos_ids for RoPE
    batch_size, seq_len, _ = in_features.shape
    pos_ids = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)
    
    return block(in_features, pos_ids=pos_ids)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    model.load_state_dict(weights)
    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    norm = RMSNorm(d_model, eps)
    norm.weight.data = weights 
    return norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    clip_gradient_norm(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return get_lr_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return load_checkpoint(model, optimizer, src)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    byte_to_id = {v: k for k, v in vocab.items()}
    merges_dict = {}
    for idx, (b1, b2) in enumerate(merges):
        new_token_bytes = b1 + b2
        if new_token_bytes in byte_to_id:
            merges_dict[(byte_to_id[b1], byte_to_id[b2])] = byte_to_id[new_token_bytes]
            
    params = BPETokenizerParams(vocab=vocab, merges=merges_dict)
    return BPETokenizer(params, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe(input_path, vocab_size, special_tokens)