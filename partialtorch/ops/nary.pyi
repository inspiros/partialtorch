from typing import Optional

from partialtorch.types import _size, MaskedPair, _MaskedPairListOrTensorList


def linalg_partial_multi_dot(tensors: _MaskedPairListOrTensorList) -> MaskedPair: ...


def partial_einsum(equation: str,
                   tensors: _MaskedPairListOrTensorList,
                   *,
                   path: Optional[_size] = None) -> MaskedPair: ...
