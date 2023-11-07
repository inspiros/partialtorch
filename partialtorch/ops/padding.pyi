from typing import Tuple, Union, Optional

from partialtorch.types import _float, _int, _bool, _size, Tensor, MaskedPair, _MaskedPairOrTensor


def pad(self: _MaskedPairOrTensor,
        pad: Union[_int, _size],
        mode: str = "constant",
        value: Optional[_float] = None,
        *,
        mask_mode: str = "constant",
        mask_value: Optional[_bool] = None) -> MaskedPair: ...
