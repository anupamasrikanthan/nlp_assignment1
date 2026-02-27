import torch
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Iterable
from typing import Any, BinaryIO, IO
import os

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(state, out)

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str | os.PathLike | BinaryIO | IO[bytes],
) -> int:
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    # Numerically stable softmax
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

def cross_entropy(
    inputs: Float[Tensor, "batch_size vocab_size"], 
    targets: Int[Tensor, "batch_size"]
) -> Float[Tensor, ""]:
    # 1. Log-sum-exp trick for the denominator (log(sum(exp(x))))
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = max_logits + torch.log(
        torch.sum(torch.exp(inputs - max_logits), dim=-1, keepdim=True)
    )
    
    # 2. Extract logits corresponding to the target classes
    # loss = log(sum(exp(logits))) - target_logit
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1))
    
    # 3. Compute average loss over the batch
    loss = log_sum_exp - target_logits
    return torch.mean(loss)

def clip_gradient_norm(
    parameters: Iterable[torch.nn.Parameter], 
    max_l2_norm: float
) -> None:
    # 1. Collect all gradients
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    
    # 2. Calculate the global L2 norm across all parameters
    # The norm of a vector formed by concatenating all gradients
    total_norm = torch.norm(
        torch.stack([torch.norm(g, 2) for g in grads]), 2
    )
    
    # 3. Apply the scaling factor if the norm exceeds max_l2_norm
    if total_norm > max_l2_norm:
        # clip_coeff = max_norm / total_norm
        clip_coeff = max_l2_norm / (total_norm + 1e-6)
        for g in grads:
            g.detach().mul_(clip_coeff)