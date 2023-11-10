from typing import Tuple, Optional

import torch.nn.modules.padding
from partialtorch.types import MaskedPair
from torch.nn.common_types import _size_2_t, _size_4_t, _size_6_t
from torch.nn.modules.utils import _pair, _quadruple, _ntuple

from .. import functional as partial_F

__all__ = ['CircularPad1d', 'CircularPad2d', 'CircularPad3d', 'ConstantPad1d', 'ConstantPad2d',
           'ConstantPad3d', 'ReflectionPad1d', 'ReflectionPad2d', 'ReflectionPad3d',
           'ReplicationPad1d', 'ReplicationPad2d', 'ReplicationPad3d', 'ZeroPad1d', 'ZeroPad2d', 'ZeroPad3d']


class _CircularPadNd(torch.nn.modules.padding._CircularPadNd):
    __constants__ = ['padding', 'mask_mode', 'mask_value']
    mask_mode: str
    mask_value: Optional[bool]

    def __init__(self, mask_mode: str, mask_value: Optional[bool]):
        super().__init__()
        self.mask_mode = mask_mode
        self.mask_value = mask_value

    def _check_input_dim(self, input: MaskedPair) -> None:
        raise NotImplementedError

    def forward(self, input: MaskedPair) -> MaskedPair:
        self._check_input_dim(input)
        return partial_F.pad(input, self.padding, 'circular', mask_mode=self.mask_mode, mask_value=self.mask_value)

    def extra_repr(self) -> str:
        res = f'{self.padding}, mask_mode={self.mask_mode}'
        if self.mask_value is not None:
            res += f', mask_value={self.mask_value}'
        return res


class CircularPad1d(_CircularPadNd):
    r"""See :class:`torch.nn.CircularPad1d` for details.
    """
    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(**factory_kwargs)
        self.padding = _pair(padding)

    def _check_input_dim(self, input: MaskedPair) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                f"expected 2D or 3D input (got {input.dim()}D input)"
            )


class CircularPad2d(_CircularPadNd):
    r"""See :class:`torch.nn.CircularPad2d` for details.
    """
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(**factory_kwargs)
        self.padding = _quadruple(padding)

    def _check_input_dim(self, input: MaskedPair) -> None:
        if input.dim() != 3 and input.dim() != 4:
            raise ValueError(
                f"expected 3D or 4D input (got {input.dim()}D input)"
            )


class CircularPad3d(_CircularPadNd):
    r"""See :class:`torch.nn.CircularPad3d` for details.
    """
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(**factory_kwargs)
        self.padding = _ntuple(6)(padding)

    def _check_input_dim(self, input: MaskedPair) -> None:
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError(
                f"expected 4D or 5D input (got {input.dim()}D input)"
            )


class _ConstantPadNd(torch.nn.modules.padding._ConstantPadNd):
    __constants__ = ['padding', 'value', 'mask_mode', 'mask_value']
    mask_mode: str
    mask_value: Optional[bool]

    def __init__(self, value: float, mask_mode: str, mask_value: Optional[bool]) -> None:
        super().__init__(value)
        self.mask_mode = mask_mode
        self.mask_value = mask_value

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.pad(input, self.padding, 'constant', self.value,
                             mask_mode=self.mask_mode, mask_value=self.mask_value)

    def extra_repr(self) -> str:
        res = f'padding={self.padding}, value={self.value}, mask_mode={self.mask_mode}'
        if self.mask_value is not None:
            res += f', mask_value={self.mask_value}'
        return res


class ConstantPad1d(_ConstantPadNd):
    r"""See :class:`torch.nn.ConstantPad1d` for details.
    """
    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t, value: float,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None):
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(value, **factory_kwargs)
        self.padding = _pair(padding)


class ConstantPad2d(_ConstantPadNd):
    r"""See :class:`torch.nn.ConstantPad2d` for details.
    """
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t, value: float,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(value, **factory_kwargs)
        self.padding = _quadruple(padding)


class ConstantPad3d(_ConstantPadNd):
    r"""See :class:`torch.nn.ConstantPad3d` for details.
    """
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t, value: float,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(value, **factory_kwargs)
        self.padding = _ntuple(6)(padding)


class _ReflectionPadNd(torch.nn.modules.padding._ReflectionPadNd):
    __constants__ = ['padding', 'mask_mode', 'mask_value']
    mask_mode: str
    mask_value: Optional[bool]

    def __init__(self, mask_mode: str, mask_value: Optional[bool]):
        super().__init__()
        self.mask_mode = mask_mode
        self.mask_value = mask_value

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.pad(input, self.padding, 'reflect',
                             mask_mode=self.mask_mode, mask_value=self.mask_value)

    def extra_repr(self) -> str:
        res = f'{self.padding}, mask_mode={self.mask_mode}'
        if self.mask_value is not None:
            res += f', mask_value={self.mask_value}'
        return res


class ReflectionPad1d(_ReflectionPadNd):
    r"""See :class:`torch.nn.ReflectionPad1d` for details.
    """
    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(**factory_kwargs)
        self.padding = _pair(padding)


class ReflectionPad2d(_ReflectionPadNd):
    r"""See :class:`torch.nn.ReflectionPad2d` for details.
    """
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(**factory_kwargs)
        self.padding = _quadruple(padding)


class ReflectionPad3d(_ReflectionPadNd):
    r"""See :class:`torch.nn.ReflectionPad3d` for details.
    """
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(**factory_kwargs)
        self.padding = _ntuple(6)(padding)


class _ReplicationPadNd(torch.nn.modules.padding._ReplicationPadNd):
    __constants__ = ['padding', 'mask_mode', 'mask_value']
    mask_mode: str
    mask_value: Optional[bool]

    def __init__(self, mask_mode: str, mask_value: Optional[bool]):
        super().__init__()
        self.mask_mode = mask_mode
        self.mask_value = mask_value

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.pad(input, self.padding, 'replicate',
                             mask_mode=self.mask_mode, mask_value=self.mask_value)

    def extra_repr(self) -> str:
        res = f'{self.padding}, mask_mode={self.mask_mode}'
        if self.mask_value is not None:
            res += f', mask_value={self.mask_value}'
        return res


class ReplicationPad1d(_ReplicationPadNd):
    r"""See :class:`torch.nn.ReplicationPad1d` for details.
    """
    padding: Tuple[int, int]

    def __init__(self, padding: _size_2_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(**factory_kwargs)
        self.padding = _pair(padding)


class ReplicationPad2d(_ReplicationPadNd):
    r"""See :class:`torch.nn.ReplicationPad2d` for details.
    """
    padding: Tuple[int, int, int, int]

    def __init__(self, padding: _size_4_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(**factory_kwargs)
        self.padding = _quadruple(padding)


class ReplicationPad3d(_ReplicationPadNd):
    r"""See :class:`torch.nn.ReplicationPad3d` for details.
    """
    padding: Tuple[int, int, int, int, int, int]

    def __init__(self, padding: _size_6_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(**factory_kwargs)
        self.padding = _ntuple(6)(padding)


class ZeroPad1d(ConstantPad1d):
    r"""See :class:`torch.nn.ZeroPad1d` for details.
    """

    def __init__(self, padding: _size_2_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(padding, 0., **factory_kwargs)

    def extra_repr(self) -> str:
        res = f'{self.padding}, mask_mode={self.mask_mode}'
        if self.mask_value is not None:
            res += f', mask_value={self.mask_value}'
        return res


class ZeroPad2d(ConstantPad2d):
    r"""See :class:`torch.nn.ZeroPad2d` for details.
    """

    def __init__(self, padding: _size_4_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(padding, 0., **factory_kwargs)

    def extra_repr(self) -> str:
        res = f'{self.padding}, mask_mode={self.mask_mode}'
        if self.mask_value is not None:
            res += f', mask_value={self.mask_value}'
        return res


class ZeroPad3d(ConstantPad3d):
    r"""See :class:`torch.nn.ZeroPad3d` for details.
    """

    def __init__(self, padding: _size_6_t,
                 mask_mode: str = 'constant',
                 mask_value: Optional[bool] = None) -> None:
        factory_kwargs = {'mask_mode': mask_mode, 'mask_value': mask_value}
        super().__init__(padding, 0., **factory_kwargs)

    def extra_repr(self) -> str:
        res = f'{self.padding}, mask_mode={self.mask_mode}'
        if self.mask_value is not None:
            res += f', mask_value={self.mask_value}'
        return res
