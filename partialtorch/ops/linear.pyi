from typing import overload, Optional

from partialtorch.types import _bool, Tensor, MaskedPair, _MaskedPairOrTensor


# linear
@overload
def partial_linear(input: _MaskedPairOrTensor, weight: Tensor, bias: Optional[Tensor] = None) -> MaskedPair: ...


@overload
def partial_linear(input: _MaskedPairOrTensor, weight: Tensor, bias: Optional[Tensor] = None, *,
                   scaled: _bool) -> MaskedPair: ...


# bilinear
@overload
def partial_bilinear(input1: _MaskedPairOrTensor, input2: _MaskedPairOrTensor,
                     weight: Tensor, bias: Optional[Tensor] = None) -> MaskedPair: ...


@overload
def partial_bilinear(input1: _MaskedPairOrTensor, input2: _MaskedPairOrTensor,
                     weight: Tensor, bias: Optional[Tensor] = None, *,
                     scaled: _bool) -> MaskedPair: ...
