import torch
import numpy as np
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, 
    batch_size: int, 
    context_length: int, 
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Determine valid starting indices.
    # We need context_length tokens for x, and the +1 token for the last y label.
    # Max index is len(dataset) - context_length - 1
    high = len(dataset) - context_length
    ix = torch.randint(high=high, size=(batch_size,))
    
    # 2. Extract sequences for x and y
    # x: dataset[i : i + context_length]
    # y: dataset[i + 1 : i + context_length + 1]
    x_list = [torch.from_numpy((dataset[i : i + context_length]).astype(np.int64)) for i in ix]
    y_list = [torch.from_numpy((dataset[i + 1 : i + context_length + 1]).astype(np.int64)) for i in ix]
    
    # 3. Stack into (batch_size, context_length) and move to device
    x = torch.stack(x_list).to(device)
    y = torch.stack(y_list).to(device)
    
    return x, y