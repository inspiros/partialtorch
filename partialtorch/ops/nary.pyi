from typing import Optional

from partialtorch.types import _size, MaskedPair, _MaskedPairListOrTensorList


def partial_einsum(equation: str, tensors: _MaskedPairListOrTensorList, *, path: Optional[_size]) -> MaskedPair: ...
