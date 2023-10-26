from typing import overload, Sequence, Optional, Union

from partialtorch.types import _int, _bool, _size, _symint, Tensor, MaskedPair, _MaskedPairOrTensor


# convolution
@overload
def partial_convolution(input: _MaskedPairOrTensor,
                        weight: Tensor,
                        bias: Optional[Tensor],
                        stride: _size,
                        padding: Sequence[_symint],
                        dilation: _size, transposed: _bool,
                        output_padding: Sequence[_symint],
                        groups: _int) -> MaskedPair: ...


@overload
def partial_convolution(input: _MaskedPairOrTensor,
                        weight: Tensor,
                        bias: Optional[Tensor],
                        stride: _size,
                        padding: Sequence[_symint],
                        dilation: _size, transposed: _bool,
                        output_padding: Sequence[_symint],
                        groups: _int,
                        *, scaled: _bool) -> MaskedPair: ...


# convnd
@overload
def partial_conv1d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: Union[_symint, Sequence[_symint]] = 0,
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1) -> MaskedPair: ...


@overload
def partial_conv1d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: str = "valid",
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1) -> MaskedPair: ...


@overload
def partial_conv1d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: Union[_symint, Sequence[_symint]] = 0,
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1,
                   *, scaled: _bool) -> MaskedPair: ...


@overload
def partial_conv1d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: str = "valid",
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1,
                   *, scaled: _bool) -> MaskedPair: ...


@overload
def partial_conv2d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: Union[_symint, Sequence[_symint]] = 0,
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1) -> MaskedPair: ...


@overload
def partial_conv2d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: str = "valid",
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1) -> MaskedPair: ...


@overload
def partial_conv2d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: Union[_symint, Sequence[_symint]] = 0,
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1,
                   *, scaled: _bool) -> MaskedPair: ...


@overload
def partial_conv2d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: str = "valid",
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1,
                   *, scaled: _bool) -> MaskedPair: ...


@overload
def partial_conv3d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: Union[_symint, Sequence[_symint]] = 0,
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1) -> MaskedPair: ...


@overload
def partial_conv3d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: str = "valid",
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1) -> MaskedPair: ...


@overload
def partial_conv3d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: Union[_symint, Sequence[_symint]] = 0,
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1,
                   *, scaled: _bool) -> MaskedPair: ...


@overload
def partial_conv3d(input: _MaskedPairOrTensor,
                   weight: Tensor,
                   bias: Optional[Tensor] = None,
                   stride: Union[_int, _size] = 1,
                   padding: str = "valid",
                   dilation: Union[_int, _size] = 1,
                   groups: _int = 1,
                   *, scaled: _bool) -> MaskedPair: ...


# conv_transposend
@overload
def partial_conv_transpose1d(input: _MaskedPairOrTensor,
                             weight: Tensor,
                             bias: Optional[Tensor] = None,
                             stride: Union[_int, _size] = 1,
                             padding: Union[_symint, Sequence[_symint]] = 0,
                             output_padding: Union[_symint, Sequence[_symint]] = 0,
                             groups: _int = 1,
                             dilation: Union[_int, _size] = 1) -> MaskedPair: ...


@overload
def partial_conv_transpose1d(input: _MaskedPairOrTensor,
                             weight: Tensor,
                             bias: Optional[Tensor] = None,
                             stride: Union[_int, _size] = 1,
                             padding: Union[_symint, Sequence[_symint]] = 0,
                             output_padding: Union[_symint, Sequence[_symint]] = 0,
                             groups: _int = 1,
                             dilation: Union[_int, _size] = 1,
                             *, scaled: _bool) -> MaskedPair: ...


@overload
def partial_conv_transpose2d(input: _MaskedPairOrTensor,
                             weight: Tensor,
                             bias: Optional[Tensor] = None,
                             stride: Union[_int, _size] = 1,
                             padding: Union[_symint, Sequence[_symint]] = 0,
                             output_padding: Union[_symint, Sequence[_symint]] = 0,
                             groups: _int = 1,
                             dilation: Union[_int, _size] = 1) -> MaskedPair: ...


@overload
def partial_conv_transpose2d(input: _MaskedPairOrTensor,
                             weight: Tensor,
                             bias: Optional[Tensor] = None,
                             stride: Union[_int, _size] = 1,
                             padding: Union[_symint, Sequence[_symint]] = 0,
                             output_padding: Union[_symint, Sequence[_symint]] = 0,
                             groups: _int = 1,
                             dilation: Union[_int, _size] = 1,
                             *, scaled: _bool) -> MaskedPair: ...


@overload
def partial_conv_transpose3d(input: _MaskedPairOrTensor,
                             weight: Tensor,
                             bias: Optional[Tensor] = None,
                             stride: Union[_int, _size] = 1,
                             padding: Union[_symint, Sequence[_symint]] = 0,
                             output_padding: Union[_symint, Sequence[_symint]] = 0,
                             groups: _int = 1,
                             dilation: Union[_int, _size] = 1) -> MaskedPair: ...


@overload
def partial_conv_transpose3d(input: _MaskedPairOrTensor,
                             weight: Tensor,
                             bias: Optional[Tensor] = None,
                             stride: Union[_int, _size] = 1,
                             padding: Union[_symint, Sequence[_symint]] = 0,
                             output_padding: Union[_symint, Sequence[_symint]] = 0,
                             groups: _int = 1,
                             dilation: Union[_int, _size] = 1,
                             *, scaled: _bool) -> MaskedPair: ...
