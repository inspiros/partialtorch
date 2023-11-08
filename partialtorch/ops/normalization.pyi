from typing import Sequence, Optional, Union

from partialtorch.types import _float, _bool, _symint, Tensor, MaskedPair, _MaskedPairOrTensor


def batch_norm(input: _MaskedPairOrTensor,
               weight: Optional[Tensor],
               bias: Optional[Tensor],
               running_mean: Optional[Tensor],
               running_var: Optional[Tensor],
               training: _bool,
               momentum: _float,
               eps: _float,
               cudnn_enabled: _bool) -> MaskedPair: ...


def instance_norm(input: _MaskedPairOrTensor,
                  weight: Optional[Tensor],
                  bias: Optional[Tensor],
                  running_mean: Optional[Tensor],
                  running_var: Optional[Tensor],
                  use_input_stats: _bool,
                  momentum: _float,
                  eps: _float,
                  cudnn_enabled: _bool) -> MaskedPair: ...


def layer_norm(input: _MaskedPairOrTensor,
               normalized_shape: Union[_symint, Sequence[_symint]],
               weight: Optional[Tensor] = None,
               bias: Optional[Tensor] = None,
               eps: _float = 1e-5,
               cudnn_enabled: _bool = True) -> MaskedPair: ...
