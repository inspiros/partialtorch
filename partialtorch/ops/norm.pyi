from typing import Optional

from partialtorch.types import Tensor, MaskedPair, _MaskedPairOrTensor


def batch_norm(input: _MaskedPairOrTensor,
               weight: Optional[Tensor],
               bias: Optional[Tensor],
               running_mean: Optional[Tensor],
               running_var: Optional[Tensor],
               training: bool,
               momentum: float,
               eps: float) -> MaskedPair: ...
