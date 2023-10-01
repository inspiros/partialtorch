from typing import overload, Optional

from partialtorch.types import _float, Number, Generator, Tensor, MaskedPair, _MaskedPairOrTensor


@overload
def rand_mask(input: _MaskedPairOrTensor,
              p: _float,
              mask_value: Optional[Number] = None,
              *,
              generator: Optional[Generator] = None) -> MaskedPair: ...


@overload
def rand_mask(input: _MaskedPairOrTensor,
              p: Tensor,
              mask_value: Optional[Number] = None,
              *,
              generator: Optional[Generator] = None) -> MaskedPair: ...


@overload
def rand_mask_(input: _MaskedPairOrTensor,
               p: _float,
               mask_value: Optional[Number] = None,
               *,
               generator: Optional[Generator] = None) -> MaskedPair: ...


@overload
def rand_mask_(input: _MaskedPairOrTensor,
               p: Tensor,
               mask_value: Optional[Number] = None,
               *,
               generator: Optional[Generator] = None) -> MaskedPair: ...
