import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.pooling import _MaxPoolNd

import partialtorch.nn.functional as partial_F
from partialtorch.types import MaskedPair

__all__ = [
    'MaskedMaxPool1d',
    'MaskedMaxPool2d',
    'MaskedMaxPool3d',
    'MaskedFractionalMaxPool2d',
    'MaskedFractionalMaxPool3d',
]


class _MaskedMaxPoolNd(_MaxPoolNd):
    pass


class MaskedMaxPool1d(_MaxPoolNd):
    r"""Mased version of :class:`torch.nn.MaxPool1d`.
    """

    kernel_size: _size_1_t
    stride: _size_1_t
    padding: _size_1_t
    dilation: _size_1_t

    def forward(self, input: MaskedPair):
        return partial_F.max_pool1d(input, self.kernel_size, self.stride,
                                    self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                    return_indices=self.return_indices)


class MaskedMaxPool2d(_MaxPoolNd):
    r"""Mased version of :class:`torch.nn.MaxPool2d`.
    """

    kernel_size: _size_2_t
    stride: _size_2_t
    padding: _size_2_t
    dilation: _size_2_t

    def forward(self, input: MaskedPair):
        return partial_F.max_pool2d(input, self.kernel_size, self.stride,
                                    self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                    return_indices=self.return_indices)


class MaskedMaxPool3d(_MaxPoolNd):
    r"""Mased version of :class:`torch.nn.MaxPool3d`.
    """

    kernel_size: _size_3_t
    stride: _size_3_t
    padding: _size_3_t
    dilation: _size_3_t

    def forward(self, input: MaskedPair):
        return partial_F.max_pool3d(input, self.kernel_size, self.stride,
                                    self.padding, self.dilation, ceil_mode=self.ceil_mode,
                                    return_indices=self.return_indices)


class MaskedFractionalMaxPool2d(nn.FractionalMaxPool2d):
    r"""Mased version of :class:`torch.nn.FractionalMaxPool2d`.
    """

    def forward(self, input: MaskedPair):
        return partial_F.fractional_max_pool2d(
            input, self.kernel_size, self.output_size, self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples)


class MaskedFractionalMaxPool3d(nn.FractionalMaxPool3d):
    r"""Mased version of :class:`torch.nn.FractionalMaxPool3d`.
    """

    def forward(self, input: MaskedPair):
        return partial_F.fractional_max_pool3d(
            input, self.kernel_size, self.output_size, self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples)
