import torch

__all__ = [
    'MaskedPair',
    'masked_pair',
]

MaskedPair = torch.classes.partialtorch.MaskedPair

# creation ops
masked_pair = torch.ops.partialtorch.masked_pair
