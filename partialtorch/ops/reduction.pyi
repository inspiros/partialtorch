from typing import overload, Tuple, Optional, Union

from partialtorch.types import (
    _float, _int, _bool, _dtype, _size, Number,
    Tensor, MaskedPair, _MaskedPairOrTensor
)


# logical
@overload
def all(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def all(self: _MaskedPairOrTensor,
        dim: Optional[Union[_int, _size]],
        keepdim: _bool = False) -> MaskedPair: ...


@overload
def any(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def any(self: _MaskedPairOrTensor,
        dim: Optional[Union[_int, _size]],
        keepdim: _bool = False) -> MaskedPair: ...


# arithmetics
@overload
def sum(self: _MaskedPairOrTensor,
        *,
        dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def sum(self: _MaskedPairOrTensor,
        dim: Optional[Union[_int, _size]],
        keepdim: _bool = False,
        *,
        dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def nansum(self: _MaskedPairOrTensor,
           *,
           dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def nansum(self: _MaskedPairOrTensor,
           dim: Optional[Union[_int, _size]],
           keepdim: _bool = False,
           *,
           dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def prod(self: _MaskedPairOrTensor,
         *,
         dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def prod(self: _MaskedPairOrTensor,
         dim: Optional[Union[_int, _size]],
         keepdim: _bool = False,
         *,
         dtype: Optional[_dtype] = None) -> MaskedPair: ...


def logsumexp(self: _MaskedPairOrTensor,
              dim: Union[_int, _size],
              keepdim: _bool = False) -> MaskedPair: ...


def softmax(self: _MaskedPairOrTensor,
            dim: _int,
            *,
            dtype: Optional[_dtype] = None) -> MaskedPair: ...


def log_softmax(self: _MaskedPairOrTensor,
                dim: _int,
                *,
                dtype: Optional[_dtype] = None) -> MaskedPair: ...


def cumsum(self: _MaskedPairOrTensor,
           dim: _int,
           *,
           dtype: Optional[_dtype] = None) -> MaskedPair: ...


def cumsum_(self: _MaskedPairOrTensor,
            dim: _int,
            *,
            dtype: Optional[_dtype] = None) -> MaskedPair: ...


def cumprod(self: _MaskedPairOrTensor,
            dim: _int,
            *,
            dtype: Optional[_dtype] = None) -> MaskedPair: ...


def cumprod_(self: _MaskedPairOrTensor,
             dim: _int,
             *,
             dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def trace(self: _MaskedPairOrTensor) -> MaskedPair: ...


# statistics
@overload
def mean(self: _MaskedPairOrTensor,
         *,
         dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def mean(self: _MaskedPairOrTensor,
         dim: Optional[Union[_int, _size]],
         keepdim: _bool = False,
         *,
         dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def nanmean(self: _MaskedPairOrTensor,
            dim: Optional[Union[_int, _size]],
            keepdim: _bool = False,
            *,
            dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def median(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def median(self: _MaskedPairOrTensor,
           dim: _int,
           keepdim: _bool = False) -> Tuple[MaskedPair, Tensor]: ...


@overload
def nanmedian(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def nanmedian(self: _MaskedPairOrTensor,
              dim: _int,
              keepdim: _bool = False) -> Tuple[MaskedPair, Tensor]: ...


@overload
def var(self: _MaskedPairOrTensor,
        unbiased: _bool = True) -> MaskedPair: ...


@overload
def var(self: _MaskedPairOrTensor,
        dim: Optional[Union[_int, _size]],
        unbiased: _bool = True,
        keepdim: _bool = False) -> MaskedPair: ...


@overload
def var(self: _MaskedPairOrTensor,
        dim: Optional[Union[_int, _size]] = None,
        correction: Optional[Number] = None,
        keepdim: _bool = False) -> MaskedPair: ...


@overload
def std(self: _MaskedPairOrTensor,
        unbiased: _bool = True) -> MaskedPair: ...


@overload
def std(self: _MaskedPairOrTensor,
        dim: Optional[Union[_int, _size]],
        unbiased: _bool = True,
        keepdim: _bool = False) -> MaskedPair: ...


@overload
def std(self: _MaskedPairOrTensor,
        dim: Optional[Union[_int, _size]] = None,
        correction: Optional[Number] = None,
        keepdim: _bool = False) -> MaskedPair: ...


def norm(self: _MaskedPairOrTensor,
         p: Optional[Number] = None,
         dim: Union[_int, _size] = (),
         keepdim: _bool = False,
         *,
         dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def linalg_norm(self: _MaskedPairOrTensor,
                ord: Number,
                dim: Optional[Union[_int, _size]] = None,
                keepdim: _bool = False,
                *,
                dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def linalg_norm(self: _MaskedPairOrTensor,
                ord: str = "fro",
                dim: Optional[Union[_int, _size]] = None,
                keepdim: _bool = False,
                *,
                dtype: Optional[_dtype] = None) -> MaskedPair: ...


def linalg_vector_norm(self: _MaskedPairOrTensor,
                       ord: Number = 2,
                       dim: Optional[Union[_int, _size]] = None,
                       keepdim: _bool = False,
                       *,
                       dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def linalg_matrix_norm(self: _MaskedPairOrTensor,
                       ord: Number,
                       dim: Union[_int, _size] = [-2, -1],
                       keepdim: _bool = False,
                       *,
                       dtype: Optional[_dtype] = None) -> MaskedPair: ...


@overload
def linalg_matrix_norm(self: _MaskedPairOrTensor,
                       ord: str = "fro",
                       dim: Union[_int, _size] = [-2, -1],
                       keepdim: _bool = False,
                       *,
                       dtype: Optional[_dtype] = None) -> MaskedPair: ...


# min max
@overload
def min(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def min(self: _MaskedPairOrTensor,
        dim: _int,
        keepdim: _bool = False) -> Tuple[MaskedPair, Tensor]: ...


@overload
def max(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def max(self: _MaskedPairOrTensor,
        dim: _int,
        keepdim: _bool = False) -> Tuple[MaskedPair, Tensor]: ...


def amin(self: _MaskedPairOrTensor,
         dim: Union[_int, _size] = [],
         keepdim: _bool = False) -> MaskedPair: ...


def amax(self: _MaskedPairOrTensor,
         dim: Union[_int, _size] = [],
         keepdim: _bool = False) -> MaskedPair: ...


def argmin(self: _MaskedPairOrTensor,
           dim: Optional[_int] = None,
           keepdim: _bool = False) -> Tensor: ...


def argmax(self: _MaskedPairOrTensor,
           dim: Optional[_int] = None,
           keepdim: _bool = False) -> Tensor: ...


def cummin(self: _MaskedPairOrTensor,
           dim: _int,
           keepdim: _bool = False) -> Tuple[MaskedPair, Tensor]: ...


def cummax(self: _MaskedPairOrTensor,
           dim: _int,
           keepdim: _bool = False) -> Tuple[MaskedPair, Tensor]: ...


# torch.nn.functional
def softmin(self: _MaskedPairOrTensor,
            dim: _int,
            *,
            dtype: Optional[_dtype] = None) -> MaskedPair: ...


def normalize(self: _MaskedPairOrTensor,
              p: Optional[Number] = 2,
              dim: _int = 1,
              eps: _float = 1e-12) -> MaskedPair: ...


# scaled arithmetics
@overload
def sum(self: _MaskedPairOrTensor,
        *,
        dtype: Optional[_dtype] = None,
        scaled: _bool) -> MaskedPair: ...


@overload
def sum(self: _MaskedPairOrTensor,
        dim: Optional[Union[_int, _size]],
        keepdim: _bool = False,
        *,
        dtype: Optional[_dtype] = None,
        scaled: _bool) -> MaskedPair: ...


@overload
def nansum(self: _MaskedPairOrTensor,
           *,
           dtype: Optional[_dtype] = None,
           scaled: _bool) -> MaskedPair: ...


@overload
def nansum(self: _MaskedPairOrTensor,
           dim: Optional[Union[_int, _size]],
           keepdim: _bool = False,
           *,
           dtype: Optional[_dtype] = None,
           scaled: _bool) -> MaskedPair: ...


@overload
def trace(self: _MaskedPairOrTensor,
          *,
          scaled: _bool) -> MaskedPair: ...
