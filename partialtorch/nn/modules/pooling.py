from typing import Optional

import torch.nn.modules.pooling
from partialtorch.types import MaskedPair
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t, _size_2_opt_t, _size_3_opt_t

from .. import functional as partial_F

__all__ = [
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'FractionalMaxPool2d', 'FractionalMaxPool3d',
    'LPPool1d', 'LPPool2d',
    'AdaptiveMaxPool1d', 'AdaptiveMaxPool2d', 'AdaptiveMaxPool3d',
    'AdaptiveAvgPool1d', 'AdaptiveAvgPool2d', 'AdaptiveAvgPool3d',
]


class _MaxPoolNd(torch.nn.modules.pooling._MaxPoolNd):
    pass


class MaxPool1d(_MaxPoolNd):
    r"""See :class:`torch.nn.MaxPool1d` for details.
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t

    def forward(self, input: MaskedPair):
        return partial_F.max_pool1d(input, self.kernel_size, self.stride,
                                    self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                    return_indices=self.return_indices)


class MaxPool2d(_MaxPoolNd):
    r"""See :class:`torch.nn.MaxPool2d` for details.
    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: MaskedPair):
        return partial_F.max_pool2d(input, self.kernel_size, self.stride,
                                    self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                    return_indices=self.return_indices)


class MaxPool3d(_MaxPoolNd):
    r"""See :class:`torch.nn.MaxPool3d` for details.
    """

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t

    def forward(self, input: MaskedPair):
        return partial_F.max_pool3d(input, self.kernel_size, self.stride,
                                    self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                    return_indices=self.return_indices)


class _AvgPoolNd(torch.nn.modules.pooling._AvgPoolNd):
    pass


class AvgPool1d(_AvgPoolNd):
    r"""See :class:`torch.nn.AvgPool1d` for details.
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(self, kernel_size: _size_1_t, stride: _size_1_t = None, padding: _size_1_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride if stride is not None else kernel_size)
        self.padding = _single(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.avg_pool1d(input, self.kernel_size, self.stride, self.padding,
                                    self.ceil_mode, self.count_include_pad)


class AvgPool2d(_AvgPoolNd):
    r"""See :class:`torch.nn.AvgPool2d` for details.
    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.avg_pool2d(input, self.kernel_size, self.stride, self.padding,
                                    self.ceil_mode, self.count_include_pad)


class AvgPool3d(_AvgPoolNd):
    r"""See :class:`torch.nn.AvgPool3d` for details.
    """

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    ceil_mode: bool
    count_include_pad: bool

    def __init__(self, kernel_size: _size_3_t, stride: Optional[_size_3_t] = None, padding: _size_3_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if (stride is not None) else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.avg_pool3d(input, self.kernel_size, self.stride, self.padding,
                                    self.ceil_mode, self.count_include_pad)

    def __setstate__(self, d):
        super().__setstate__(d)
        self.__dict__.setdefault('padding', 0)
        self.__dict__.setdefault('ceil_mode', False)
        self.__dict__.setdefault('count_include_pad', True)


class FractionalMaxPool2d(torch.nn.modules.pooling.FractionalMaxPool2d):
    r"""See :class:`torch.nn.FractionalMaxPool2d` for details.
    """

    def forward(self, input: MaskedPair):
        return partial_F.fractional_max_pool2d(
            input, self.kernel_size, self.output_size, self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples)


class FractionalMaxPool3d(torch.nn.modules.pooling.FractionalMaxPool3d):
    r"""See :class:`torch.nn.FractionalMaxPool3d` for details.
    """

    def forward(self, input: MaskedPair):
        return partial_F.fractional_max_pool3d(
            input, self.kernel_size, self.output_size, self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples)


class _LPPoolNd(torch.nn.modules.pooling._LPPoolNd):
    pass


class LPPool1d(_LPPoolNd):
    r"""See :class:`torch.nn.LPPool1d` for details.
    """

    kernel_size: _size_1_t
    stride: _size_1_t

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.lp_pool1d(input, float(self.norm_type), self.kernel_size,
                                   self.stride, self.ceil_mode)


class LPPool2d(_LPPoolNd):
    r"""See :class:`torch.nn.LPPool2d` for details.
    """

    kernel_size: _size_2_t
    stride: _size_2_t

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.lp_pool2d(input, float(self.norm_type), self.kernel_size,
                                   self.stride, self.ceil_mode)


class _AdaptiveMaxPoolNd(torch.nn.modules.pooling._AdaptiveMaxPoolNd):
    pass


class AdaptiveMaxPool1d(_AdaptiveMaxPoolNd):
    r"""See :class:`torch.nn.AdaptiveMaxPool1d` for details.
    """

    output_size: _size_1_t

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.adaptive_max_pool1d(input, self.output_size, self.return_indices)


class AdaptiveMaxPool2d(_AdaptiveMaxPoolNd):
    r"""See :class:`torch.nn.AdaptiveMaxPool2d` for details.
    """

    output_size: _size_2_t

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.adaptive_max_pool2d(input, self.output_size, self.return_indices)


class AdaptiveMaxPool3d(_AdaptiveMaxPoolNd):
    r"""See :class:`torch.nn.AdaptiveMaxPool3d` for details.
    """

    output_size: _size_3_t

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.adaptive_max_pool3d(input, self.output_size, self.return_indices)


class _AdaptiveAvgPoolNd(torch.nn.modules.pooling._AdaptiveAvgPoolNd):
    pass


class AdaptiveAvgPool1d(_AdaptiveAvgPoolNd):
    r"""See :class:`torch.nn.AdaptiveAvgPool1d` for details.
    """

    output_size: _size_1_t

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.adaptive_avg_pool1d(input, self.output_size)


class AdaptiveAvgPool2d(_AdaptiveAvgPoolNd):
    r"""See :class:`torch.nn.AdaptiveAvgPool2d` for details.
    """

    output_size: _size_2_opt_t

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.adaptive_avg_pool2d(input, self.output_size)


class AdaptiveAvgPool3d(_AdaptiveAvgPoolNd):
    r"""See :class:`torch.nn.AdaptiveAvgPool3d` for details.
    """

    output_size: _size_3_opt_t

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.adaptive_avg_pool3d(input, self.output_size)
