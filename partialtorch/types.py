from typing import Sequence, Union

from torch import Tensor, Size, Generator, memory_format as _memory_format
from torch.types import (
    _int, _float, _bool, _dtype, _device, _qscheme, _size, _layout, _dispatchkey,
    _TensorOrTensors, SymInt, Number, Device, Storage
)

from ._masked_pair import MaskedPair

__all__ = [
    '_int',
    '_float',
    '_bool',
    '_dtype',
    '_device',
    '_qscheme',
    '_size',
    '_layout',
    '_memory_format',
    '_dispatchkey',
    '_dimname',
    '_symint',
    'Size',
    'SymInt',
    'Number',
    'Device',
    'Storage',
    'Generator',
    'Tensor',
    '_TensorOrTensors',
    'MaskedPair',
    '_MaskedPairOrMaskedPairs',
    '_MaskedPairOrTensor',
    '_MaskedPairListOrTensorList',
]

_dimname = Union[str, type(Ellipsis), None]
_symint = Union[_int, SymInt]
_MaskedPairOrMaskedPairs = Union[MaskedPair, Sequence[MaskedPair]]
_MaskedPairOrTensor = Union[MaskedPair, Tensor]
_MaskedPairListOrTensorList = Union[_MaskedPairOrMaskedPairs, _TensorOrTensors]
