import torch
from torch import Tensor

def softmax(x: Tensor, dim: int) -> Tensor:
    """
    Apply numerically stable softmax to the specified dimension.
    """
    # 1. Subtract the max value for numerical stability
    max_val = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_val)
    
    # 2. Divide by the sum of exponentials
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)