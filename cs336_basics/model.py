import torch
import torch.nn as nn
import math
from typing import Optional
from cs336_basics.extra_stuff import silu, softmax

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features,in_features), device=device, dtype=dtype))
        std = math.sqrt(2.0/(in_features+out_features))
        nn.init.trunc_normal_(self.weight,mean=0.0, std=std,a=-3*std,b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight.t())

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_fp32**2, dim=-1, keepdim=True)+self.eps)
        out = (x_fp32 / rms)*self.weight.to(torch.float32)
        return out.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k
        inv_freq = 1.0/(theta**(torch.arange(0,d_k,2,device=device).float()/d_k))
        positions = torch.arange(max_seq_len,device=device).float()
        angles = torch.einsum('i,j->ij',positions, inv_freq)
        self.register_buffer('cos', torch.cos(angles),persistent=False)
        self.register_buffer('sin', torch.sin(angles),persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]
        x_reshaped = x.view(*x.shape[:-1],self.d_k//2,2)
        x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]
        x0_rot = x0*cos_pos-x1*sin_pos
        x1_rot = x0*sin_pos+x1*cos_pos
        return torch.stack((x0_rot, x1_rot), dim=-1).view_as(x)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==False, float('-inf'))   
    attn_weights = softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, rope_theta: float = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        if max_seq_len is not None and rope_theta is not None:
            self.rope = RotaryPositionalEmbedding(self.d_k, rope_theta, max_seq_len)
        else:
            self.rope = None
            
    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        K = self.k_proj(x).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        V = self.v_proj(x).view(batch_size,seq_len,self.num_heads,self.d_v).transpose(1,2)
        if self.rope is not None and pos_ids is not None:
            pos_ids_expanded = pos_ids.unsqueeze(1).expand(-1, self.num_heads, -1)
            Q = self.rope(Q,pos_ids_expanded)
            K = self.rope(K,pos_ids_expanded)
        mask = torch.tril(torch.ones((seq_len,seq_len),device=x.device,dtype=torch.bool))
        out = scaled_dot_product_attention(Q,K,V,mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self,d_model: int,num_heads: int,d_ff: int,max_seq_len: int = None,rope_theta: float = None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model,num_heads,max_seq_len,rope_theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model,d_ff)
        
    def forward(self,x: torch.Tensor,pos_ids: torch.Tensor = None) -> torch.Tensor:
        x = x+self.attn(self.ln1(x), pos_ids)
        x = x+self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self,vocab_size: int,context_length: int,d_model: int,num_layers: int,num_heads: int,d_ff: int,rope_theta: float):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size,d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta)for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model,vocab_size)
        
    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_indices.shape
        x = self.token_embeddings(in_indices)
        pos_ids = torch.arange(seq_len, device=in_indices.device).unsqueeze(0).expand(batch_size,-1)
        for layer in self.layers:
            x = layer(x,pos_ids) 
        x = self.ln_final(x)
        return self.lm_head(x)