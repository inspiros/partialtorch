from typing import Optional

from partialtorch.types import (
    _int, _bool, Tensor
)

_has_cuda: _bool
_cuda_version: _int


class MaskedPair:
    r"""pybind11 trampoline class of :class:`partialtorch.MaskedPair`.

    Notes:
        It seems to have no use at the moment because `torch.jit.script` functions
        do not accept it as argument.
    """
    data: Tensor
    mask: Optional[Tensor]
