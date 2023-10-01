"""
Modified from https://github.com/NVIDIA/partialconv/blob/master/models/partialconv2d.py
"""
from typing import List, Optional, Union

from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.conv import _ConvNd, _ConvTransposeNd
from torch.nn.modules.utils import _single, _pair, _triple

import partialtorch.nn.functional as partial_F
from partialtorch.types import Tensor, MaskedPair

__all__ = [
    'PartialConv1d',
    'PartialConv2d',
    'PartialConv3d',
    'PartialConvTranspose1d',
    'PartialConvTranspose2d',
    'PartialConvTranspose3d',
]


################################################################################
# PartialConv
################################################################################
# noinspection DuplicatedCode,PyMethodOverriding
class _PartialConvNd(_ConvNd):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_any_t,
                 stride: _size_any_t,
                 padding: Union[_size_any_t, str],
                 dilation: _size_any_t,
                 transposed: bool,
                 output_padding: Union[_size_any_t, str],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 scaled: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode, **factory_kwargs)
        self.scaled = scaled

    def extra_repr(self) -> str:
        return super().extra_repr() + 'scaled={}'.format(self.scaled)

    def _conv_forward(self,
                      input: MaskedPair,
                      weight: Tensor,
                      bias: Optional[Tensor]) -> MaskedPair:
        raise NotImplementedError

    def forward(self, input: MaskedPair) -> MaskedPair:
        if self.padding_mode != 'zeros':
            raise ValueError(f'Only `zeros` padding mode is supported.')
        return self._conv_forward(input, self.weight, self.bias)


class PartialConv1d(_PartialConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: Union[str, _size_1_t] = 0,
                 dilation: _size_1_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 scaled: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, scaled, **factory_kwargs)

    def _conv_forward(self,
                      input: MaskedPair,
                      weight: Tensor,
                      bias: Optional[Tensor]) -> MaskedPair:
        return partial_F.partial_conv1d(input, weight, bias, self.stride,
                                        self.padding, self.dilation, self.groups,
                                        scaled=self.scaled)


class PartialConv2d(_PartialConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 scaled: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, scaled, **factory_kwargs)

    def _conv_forward(self,
                      input: MaskedPair,
                      weight: Tensor,
                      bias: Optional[Tensor]) -> MaskedPair:
        return partial_F.partial_conv2d(input, weight, bias, self.stride,  # type: ignore[has-type]
                                        self.padding, self.dilation, self.groups,
                                        scaled=self.scaled)


class PartialConv3d(_PartialConvNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 padding: Union[str, _size_3_t] = 0,
                 dilation: _size_3_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 scaled: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _triple(0), groups, bias, padding_mode, scaled, **factory_kwargs)

    def _conv_forward(self,
                      input: MaskedPair,
                      weight: Tensor,
                      bias: Optional[Tensor]) -> MaskedPair:
        return partial_F.partial_conv3d(input, weight, bias, self.stride,  # type: ignore[has-type]
                                        self.padding, self.dilation, self.groups,
                                        scaled=self.scaled)


################################################################################
# PartialConvTransposed
################################################################################
# noinspection DuplicatedCode,PyMethodOverriding
class _PartialConvTransposeNd(_ConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_any_t,
                 stride: _size_any_t,
                 padding: Union[str, _size_any_t],
                 dilation: _size_any_t,
                 transposed: bool,
                 output_padding: _size_any_t,
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 scaled: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, transposed, output_padding,
                         groups, bias, padding_mode, device, dtype)
        self.scaled = scaled

    def extra_repr(self) -> str:
        return super().extra_repr() + 'scaled={}'.format(self.scaled)

    def _conv_transpose_forward(self,
                                input: MaskedPair,
                                weight: Tensor,
                                bias: Optional[Tensor],
                                output_padding: List[int]) -> MaskedPair:
        raise NotImplementedError

    def forward(self,
                input: MaskedPair,
                output_size: Optional[List[int]] = None) -> MaskedPair:
        if self.padding_mode != 'zeros':
            raise ValueError(f'Only `zeros` padding mode is supported.')

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = self.weight.ndim - 2
        output_padding = self._output_padding(
            input.data, output_size, self.stride, self.padding, self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims, self.dilation)  # type: ignore[arg-type]

        return self._conv_transpose_forward(input, self.weight, self.bias, output_padding)


class PartialConvTranspose1d(_PartialConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_1_t,
                 stride: _size_1_t = 1,
                 padding: _size_1_t = 0,
                 output_padding: _size_1_t = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: _size_1_t = 1,
                 padding_mode: str = 'zeros',
                 scaled: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        output_padding = _single(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         True, output_padding, groups, bias, padding_mode, scaled, **factory_kwargs)

    def _conv_transpose_forward(self,
                                input: MaskedPair,
                                weight: Tensor,
                                bias: Optional[Tensor],
                                output_padding: _size_1_t) -> MaskedPair:
        return partial_F.partial_conv_transpose1d(
            input, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation, scaled=self.scaled)


class PartialConvTranspose2d(_PartialConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: _size_2_t = 0,
                 output_padding: _size_2_t = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: _size_2_t = 1,
                 padding_mode: str = 'zeros',
                 scaled: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         True, output_padding, groups, bias, padding_mode, scaled, **factory_kwargs)

    def _conv_transpose_forward(self,
                                input: MaskedPair,
                                weight: Tensor,
                                bias: Optional[Tensor],
                                output_padding: _size_2_t) -> MaskedPair:
        return partial_F.partial_conv_transpose2d(
            input, weight, bias, self.stride, self.padding,  # type: ignore[has-type]
            output_padding, self.groups, self.dilation, scaled=self.scaled)


class PartialConvTranspose3d(_PartialConvTransposeNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_3_t,
                 stride: _size_3_t = 1,
                 padding: _size_3_t = 0,
                 output_padding: _size_3_t = 0,
                 groups: int = 1,
                 bias: bool = True,
                 dilation: _size_3_t = 1,
                 padding_mode: str = 'zeros',
                 scaled: bool = True,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        output_padding = _triple(output_padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         True, output_padding, groups, bias, padding_mode, scaled, **factory_kwargs)

    def _conv_transpose_forward(self,
                                input: MaskedPair,
                                weight: Tensor,
                                bias: Optional[Tensor],
                                output_padding: _size_3_t) -> MaskedPair:
        return partial_F.partial_conv_transpose3d(
            input, weight, bias, self.stride, self.padding,  # type: ignore[has-type]
            output_padding, self.groups, self.dilation, scaled=self.scaled)
