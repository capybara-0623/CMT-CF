import random
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int):
    mask = mask.float()
    return (x * mask).sum(dim=dim) / (mask.sum(dim=dim).clamp(min=1.0))

def sequence_mask(lengths: torch.Tensor, max_len=None):
    if max_len is None:
        max_len = int(lengths.max().item())
    arange = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return arange < lengths.unsqueeze(1)