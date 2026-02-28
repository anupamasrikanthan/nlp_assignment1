import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta2: {betas[1]}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['step'] += 1
                m, v = state['m'], state['v']
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).add_(grad.pow(2), alpha=1-beta2)
                step = state['step']
                bias_correction1 = 1-beta1**step
                bias_correction2 = 1-beta2**step
                step_size = group['lr']*(bias_correction2**0.5)/bias_correction1
                denom = v.sqrt().add_(group['eps'])
                p.addcdiv_(m,denom,value=-step_size)
                if group['weight_decay']>0:
                    p.add_(p, alpha=-group['lr']*group['weight_decay'])
        return loss

def get_lr_cosine_schedule(
it: int,
max_learning_rate: float,
min_learning_rate: float,
warmup_iters: int,
cosine_cycle_iters: int,
) -> float:
    if it<warmup_iters:
        return max_learning_rate*it/warmup_iters
    if it>cosine_cycle_iters:
        return min_learning_rate
    decay_ratio=(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_learning_rate+coeff*(max_learning_rate-min_learning_rate)