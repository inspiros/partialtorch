from typing import Sequence, Tuple, Union

from partialtorch.types import _int, _bool, _size, _symint, Tensor, MaskedPair, _MaskedPairOrTensor


# avg_pool
def avg_pool1d(input: _MaskedPairOrTensor,
               kernel_size: Union[_int, _size],
               stride: Union[_int, _size] = 1,
               padding: Union[_int, _size] = 0,
               ceil_mode: _bool = False,
               count_include_pad: _bool = True) -> MaskedPair: ...


def avg_pool2d(input: _MaskedPairOrTensor,
               kernel_size: Union[_int, _size],
               stride: Union[_int, _size] = 1,
               padding: Union[_int, _size] = 0,
               ceil_mode: _bool = False,
               count_include_pad: _bool = True) -> MaskedPair: ...


def avg_pool3d(input: _MaskedPairOrTensor,
               kernel_size: Union[_int, _size],
               stride: Union[_int, _size] = 1,
               padding: Union[_int, _size] = 0,
               ceil_mode: _bool = False,
               count_include_pad: _bool = True) -> MaskedPair: ...


def adaptive_avg_pool1d(input: _MaskedPairOrTensor,
                        output_size: Union[_int, _size]) -> MaskedPair: ...


def adaptive_avg_pool2d(input: _MaskedPairOrTensor,
                        output_size: Union[_symint, Sequence[_symint]]) -> MaskedPair: ...


def adaptive_avg_pool3d(input: _MaskedPairOrTensor,
                        output_size: Union[_symint, Sequence[_symint]]) -> MaskedPair: ...


# max_pool
def max_pool1d(input: _MaskedPairOrTensor,
               kernel_size: Union[_int, _size],
               stride: Union[_int, _size] = 1,
               padding: Union[_int, _size] = 0,
               dilation: Union[_int, _size] = 1,
               ceil_mode: _bool = False) -> MaskedPair: ...


def max_pool2d(input: _MaskedPairOrTensor,
               kernel_size: Union[_int, _size],
               stride: Union[_int, _size] = 1,
               padding: Union[_int, _size] = 0,
               dilation: Union[_int, _size] = 1,
               ceil_mode: _bool = False) -> MaskedPair: ...


def max_pool3d(input: _MaskedPairOrTensor,
               kernel_size: Union[_int, _size],
               stride: Union[_int, _size] = 1,
               padding: Union[_int, _size] = 0,
               dilation: Union[_int, _size] = 1,
               ceil_mode: _bool = False) -> MaskedPair: ...


def max_pool1d_with_indices(input: _MaskedPairOrTensor,
                            kernel_size: Union[_int, _size],
                            stride: Union[_int, _size] = 1,
                            padding: Union[_int, _size] = 0,
                            dilation: Union[_int, _size] = 1,
                            ceil_mode: _bool = False) -> Tuple[MaskedPair, Tensor]: ...


def max_pool2d_with_indices(input: _MaskedPairOrTensor,
                            kernel_size: Union[_int, _size],
                            stride: Union[_int, _size] = 1,
                            padding: Union[_int, _size] = 0,
                            dilation: Union[_int, _size] = 1,
                            ceil_mode: _bool = False) -> Tuple[MaskedPair, Tensor]: ...


def max_pool3d_with_indices(input: _MaskedPairOrTensor,
                            kernel_size: Union[_int, _size],
                            stride: Union[_int, _size] = 1,
                            padding: Union[_int, _size] = 0,
                            dilation: Union[_int, _size] = 1,
                            ceil_mode: _bool = False) -> Tuple[MaskedPair, Tensor]: ...


def adaptive_max_pool1d(input: _MaskedPairOrTensor,
                        output_size: Union[_int, _size]) -> Tuple[MaskedPair, Tensor]: ...


def adaptive_max_pool2d(input: _MaskedPairOrTensor,
                        output_size: Union[_int, _size]) -> Tuple[MaskedPair, Tensor]: ...


def adaptive_max_pool3d(input: _MaskedPairOrTensor,
                        output_size: Union[_int, _size]) -> Tuple[MaskedPair, Tensor]: ...


# fractional_max_pool
def fractional_max_pool2d(input: _MaskedPairOrTensor,
                          kernel_size: Union[_int, _size],
                          output_size: Union[_int, _size],
                          random_sample: Tensor) -> Tuple[MaskedPair, Tensor]: ...


def fractional_max_pool3d(input: _MaskedPairOrTensor,
                          kernel_size: Union[_int, _size],
                          output_size: Union[_int, _size],
                          random_sample: Tensor) -> Tuple[MaskedPair, Tensor]: ...
