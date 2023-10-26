from typing import Optional

from partialtorch.types import _float, _bool, Tensor, MaskedPair, _MaskedPairOrTensor


def batch_norm(input: _MaskedPairOrTensor,
               weight: Optional[Tensor],
               bias: Optional[Tensor],
               running_mean: Optional[Tensor],
               running_var: Optional[Tensor],
               training: _bool,
               momentum: _float,
               eps: _float) -> MaskedPair: ...
