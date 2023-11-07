import torch.nn as nn
import torch.nn.modules.pooling
from partialtorch.types import MaskedPair
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from .. import functional as partial_F

__all__ = [
    'MaxPool1d',
    'MaxPool2d',
    'MaxPool3d',
    'FractionalMaxPool2d',
    'FractionalMaxPool3d',
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


class FractionalMaxPool2d(nn.FractionalMaxPool2d):
    r"""See :class:`torch.nn.FractionalMaxPool2d` for details.
    """

    def forward(self, input: MaskedPair):
        return partial_F.fractional_max_pool2d(
            input, self.kernel_size, self.output_size, self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples)


class FractionalMaxPool3d(nn.FractionalMaxPool3d):
    r"""See :class:`torch.nn.FractionalMaxPool3d` for details.
    """

    def forward(self, input: MaskedPair):
        return partial_F.fractional_max_pool3d(
            input, self.kernel_size, self.output_size, self.output_ratio,
            self.return_indices,
            _random_samples=self._random_samples)
