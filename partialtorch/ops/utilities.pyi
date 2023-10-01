from typing import Optional

from partialtorch.types import (
    _bool, Tensor,
    _MaskedPairOrTensor, _MaskedPairListOrTensorList
)


def _backward(self: _MaskedPairOrTensor,
              inputs: _MaskedPairListOrTensorList,
              gradient: Optional[Tensor] = None,
              retain_graph: Optional[_bool] = None,
              create_graph: bool = False) -> None: ...
