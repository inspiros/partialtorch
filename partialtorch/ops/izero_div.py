import torch

__all__ = [
    '_izero_div',
    '_izero_div_',
    '_izero_ldiv',
    '_izero_ldiv_',
]

_izero_div = torch.ops.partialtorch._izero_div
_izero_div_ = torch.ops.partialtorch._izero_div_
_izero_ldiv = torch.ops.partialtorch._izero_ldiv
_izero_ldiv_ = torch.ops.partialtorch._izero_ldiv_
