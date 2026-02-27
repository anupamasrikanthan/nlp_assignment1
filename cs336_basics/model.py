import torch

import torch.nn as nn
from jaxtyping import Float, Int, Bool
from torch import Tensor
from typing import Optional

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Numerically stable softmax implementation.
    """
    # Subtract max for numerical stability to prevent overflow
    max_val = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # Rename self.W to self.weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

    def forward(self, x):
        return torch.einsum("...i, oi -> ...o", x, self.weight)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # Rename self.g to self.weight
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + self.eps)
        output = (x_f32 / rms) * self.weight
        return output.to(in_dtype)

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # Ensure this is named self.weight
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return self.weight[token_ids]
import math

def scaled_dot_product_attention(
    Q: Tensor, 
    K: Tensor, 
    V: Tensor, 
    mask: Tensor | None = None
) -> Tensor:
    """
    Compute Scaled Dot-Product Attention.
    Q, K: (..., queries/keys, d_k)
    V: (..., values, d_v)
    """
    d_k = Q.size(-1)
    
    # 1. Compute scores: (Q * K^T) / sqrt(d_k)
    # Using einsum to handle arbitrary batch dimensions safely
    scores = torch.einsum("...qd, ...kd -> ...qk", Q, K) / math.sqrt(d_k)
    
    # 2. Apply optional mask
    if mask is not None:
        # Replace False/0 in the mask with -infinity before softmax
        scores = scores.masked_fill(mask == False, float("-inf"))
    
    # 3. Apply softmax to get attention weights
    weights = softmax(scores, dim=-1)
    
    # 4. Compute weighted sum of values: weights * V
    return torch.einsum("...qk, ...kv -> ...qv", weights, V)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        # Using the custom Linear class you built previously
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        # Implementation of SiLU(x) = x * sigmoid(x)
        # SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) * W3x)
        x_w1 = self.w1(x)
        silu_out = x_w1 * torch.sigmoid(x_w1) 
        return self.w2(silu_out * self.w3(x))
    

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope_theta=10000.0, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        
        # Initialize RoPE
        self.rope = RotaryPositionalEmbedding(self.d_k, theta=rope_theta)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 1. Project and split heads: (batch, num_heads, seq_len, d_k)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. Apply RoPE to Q and K
        # Create positions: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        # 3. Scaled Dot-Product Attention with Causal Mask
        attn_scores = torch.einsum("bnqd, bnkd -> bnqk", q, k) / math.sqrt(self.d_k)
        
        indices = torch.arange(seq_len, device=x.device)
        mask = indices.view(1, -1) > indices.view(-1, 1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = softmax(attn_scores, dim=-1)
        out = torch.einsum("bnqk, bnkv -> bnqv", attn_weights, v)

        # 4. Recombine and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_proj(out)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        self.d_k = d_k
        # Precompute the rotation frequencies (theta_i)
        # theta_i = theta ^ (-2(i-1)/d) for i = 1, 2, ..., d/2
        indices = torch.arange(0, d_k, 2).float()
        inv_freq = 1.0 / (theta ** (indices / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x shape: (..., seq_len, d_k)
        # token_positions shape: (..., seq_len)
        
        # Calculate angles: (..., seq_len, d_k/2)
        angles = torch.einsum("...s, j -> ...sj", token_positions.float(), self.inv_freq)
        
        # Calculate cos and sin components
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # Repeat components for element-wise rotation logic
        # [cos, cos, sin, sin] pattern for rotation matrix
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        
        # Apply the rotation: x_rotated = x * cos + x_interleaved * sin
        # x_interleaved maps [x1, x2, x3, x4] to [-x2, x1, -x4, x3]
        x_interleaved = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
        
        return x * cos + x_interleaved * sin
    

    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # Pre-norm architecture: Norm -> Sublayer -> Add
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # Residual connections around each sub-layer
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta=10000.0, device=None, dtype=None):
        super().__init__()
        # token_embeddings must be your custom class, not nn.Embedding
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        # ModuleList is permitted for stacking layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # lm_head must be your custom Linear class
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, "batch seq"]):
        h = self.token_embeddings(x)
        for layer in self.layers:
            h = layer(h)
        # Apply final norm then project to logits
        return self.lm_head(self.ln_final(h))