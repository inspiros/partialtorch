from typing import overload, List, Sequence, Optional, Union

import torch

from partialtorch.types import (
    _int, _bool, _dtype, _layout, _device, _size, _memory_format, _dimname, _symint,
    Tensor, MaskedPair, _MaskedPairOrTensor, _MaskedPairListOrTensorList
)


# to
@overload
def to(self: _MaskedPairOrTensor,
       dtype: Optional[_dtype] = None,
       layout: Optional[_layout] = None,
       device: Optional[_device] = None,
       pin_memory: Optional[_bool] = None,
       non_blocking: _bool = False,
       copy: _bool = False,
       memory_format: Optional[_memory_format] = None) -> MaskedPair: ...


@overload
def to(self: _MaskedPairOrTensor,
       device: _device,
       dtype: _dtype,
       non_blocking: _bool = False,
       copy: _bool = False,
       memory_format: Optional[_memory_format] = None) -> MaskedPair: ...


@overload
def to(self: _MaskedPairOrTensor,
       dtype: _dtype,
       non_blocking: _bool = False,
       copy: _bool = False,
       memory_format: Optional[_memory_format] = None) -> MaskedPair: ...


@overload
def to(self: _MaskedPairOrTensor,
       other: _MaskedPairOrTensor,
       non_blocking: _bool = False,
       copy: _bool = False,
       memory_format: Optional[_memory_format] = None) -> MaskedPair: ...


def cpu(self: _MaskedPairOrTensor) -> MaskedPair: ...


def cuda(self: _MaskedPairOrTensor) -> MaskedPair: ...


# one to one
def clone(self: _MaskedPairOrTensor,
          *,
          memory_format: Optional[_memory_format] = None) -> MaskedPair: ...


def contiguous(self: _MaskedPairOrTensor,
               *,
               memory_format: _memory_format = torch.contiguous_format) -> MaskedPair: ...


def detach(self: _MaskedPairOrTensor) -> MaskedPair: ...


def detach_copy(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def atleast_1d(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def atleast_2d(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def atleast_3d(self: _MaskedPairOrTensor) -> MaskedPair: ...


def diag(self: _MaskedPairOrTensor,
         diagonal: _int = 0) -> MaskedPair: ...


def diag_embed(self: _MaskedPairOrTensor,
               offset: _int = 0,
               dim1: _int = 0,
               dim2: _int = 1) -> MaskedPair: ...


def diagflat(self: _MaskedPairOrTensor,
             offset: _int = 0) -> MaskedPair: ...


@overload
def diagonal(self: _MaskedPairOrTensor,
             offset: _int = 0,
             dim1: _int = 0,
             dim2: _int = 1) -> MaskedPair: ...


@overload
def diagonal(self: _MaskedPairOrTensor,
             *,
             outdim: _dimname,
             dim1: _dimname,
             dim2: _dimname,
             offset: _int = 0) -> MaskedPair: ...


def linalg_diagonal(self: _MaskedPairOrTensor,
                    offset: _int = 0,
                    dim1: _int = -2,
                    dim2: _int = -1) -> MaskedPair: ...


@overload
def narrow(self: _MaskedPairOrTensor,
           dim: _int,
           start: _symint,
           length: _symint) -> MaskedPair: ...


@overload
def narrow(self: _MaskedPairOrTensor,
           dim: _int,
           start: Tensor,
           length: _symint) -> MaskedPair: ...


def narrow_copy(self: _MaskedPairOrTensor,
                dim: _int,
                start: _symint,
                length: _symint) -> MaskedPair: ...


@overload
def select(self: _MaskedPairOrTensor,
           dim: _int,
           index: _symint) -> MaskedPair: ...


@overload
def select(self: _MaskedPairOrTensor,
           dim: _dimname,
           index: _symint) -> MaskedPair: ...


def repeat(self: _MaskedPairOrTensor,
           repeats: Sequence[_symint]) -> MaskedPair: ...


@overload
def repeat_interleave(self: _MaskedPairOrTensor,
                      repeats: _symint,
                      dim: Optional[_int] = None,
                      *,
                      output_size: Optional[_int] = None) -> MaskedPair: ...


@overload
def repeat_interleave(self: _MaskedPairOrTensor,
                      repeats: Tensor,
                      dim: Optional[_int] = None,
                      *,
                      output_size: Optional[_int] = None) -> MaskedPair: ...


def tile(self: _MaskedPairOrTensor,
         dims: Sequence[_symint]) -> MaskedPair: ...


def ravel(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def flatten(self: _MaskedPairOrTensor,
            start_dim: _int = 0,
            end_dim: _int = -1) -> MaskedPair: ...


@overload
def flatten(self: _MaskedPairOrTensor,
            start_dim: _int,
            end_dim: _int,
            out_dim: _dimname) -> MaskedPair: ...


@overload
def flatten(self: _MaskedPairOrTensor,
            start_dim: _dimname,
            end_dim: _dimname,
            out_dim: _dimname) -> MaskedPair: ...


@overload
def flatten(self: _MaskedPairOrTensor,
            dims: Sequence[_dimname],
            out_dim: _dimname) -> MaskedPair: ...


@overload
def unflatten(self: _MaskedPairOrTensor,
              dim: _int,
              sizes: Sequence[_symint]) -> MaskedPair: ...


@overload
def unflatten(self: _MaskedPairOrTensor,
              dim: _dimname,
              sizes: Sequence[_symint],
              names: Sequence[_dimname]) -> MaskedPair: ...


def broadcast_to(self: _MaskedPairOrTensor,
                 size: Sequence[_symint]) -> MaskedPair: ...


def expand(self: _MaskedPairOrTensor,
           size: Sequence[_symint],
           *,
           implicit: _bool = False) -> MaskedPair: ...


def expand_as(self: _MaskedPairOrTensor,
              other: _MaskedPairOrTensor) -> MaskedPair: ...


def reshape(self: _MaskedPairOrTensor,
            size: Sequence[_symint]) -> MaskedPair: ...


def reshape_as(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def view(self: _MaskedPairOrTensor,
         size: Sequence[_symint]) -> MaskedPair: ...


@overload
def view(self: _MaskedPairOrTensor,
         dtype: _dtype) -> MaskedPair: ...


def view_as(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def squeeze(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def squeeze_(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def squeeze(self: _MaskedPairOrTensor,
            dim: _int) -> MaskedPair: ...


@overload
def squeeze(self: _MaskedPairOrTensor,
            dim: _dimname) -> MaskedPair: ...


@overload
def squeeze(self: _MaskedPairOrTensor,
            dim: Union[_int, _size]) -> MaskedPair: ...


def unsqueeze(self: _MaskedPairOrTensor,
              dim: _int) -> MaskedPair: ...


def unsqueeze_(self: _MaskedPairOrTensor,
               dim: _int) -> MaskedPair: ...


def matrix_H(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def moveaxis(self: _MaskedPairOrTensor,
             source: Union[_int, _size],
             destination: Union[_int, _size]) -> MaskedPair: ...


@overload
def moveaxis(self: _MaskedPairOrTensor,
             source: _int,
             destination: _int) -> MaskedPair: ...


@overload
def moveaxes(self: _MaskedPairOrTensor,
             source: Union[_int, _size],
             destination: Union[_int, _size]) -> MaskedPair: ...


@overload
def moveaxes(self: _MaskedPairOrTensor,
             source: _int,
             destination: _int) -> MaskedPair: ...


@overload
def movedim(self: _MaskedPairOrTensor,
            source: Union[_int, _size],
            destination: Union[_int, _size]) -> MaskedPair: ...


@overload
def movedim(self: _MaskedPairOrTensor,
            source: _int,
            destination: _int) -> MaskedPair: ...


@overload
def movedims(self: _MaskedPairOrTensor,
             source: Union[_int, _size],
             destination: Union[_int, _size]) -> MaskedPair: ...


@overload
def movedims(self: _MaskedPairOrTensor,
             source: _int,
             destination: _int) -> MaskedPair: ...


def swapaxis(self: _MaskedPairOrTensor,
             axis0: _int,
             axis1: _int) -> MaskedPair: ...


def swapaxis_(self: _MaskedPairOrTensor,
              axis0: _int,
              axis1: _int) -> MaskedPair: ...


def swapaxes(self: _MaskedPairOrTensor,
             axis0: _int,
             axis1: _int) -> MaskedPair: ...


def swapaxes_(self: _MaskedPairOrTensor,
              axis0: _int,
              axis1: _int) -> MaskedPair: ...


def swapdim(self: _MaskedPairOrTensor,
            dim0: _int,
            dim1: _int) -> MaskedPair: ...


def swapdim_(self: _MaskedPairOrTensor,
             dim0: _int,
             dim1: _int) -> MaskedPair: ...


def swapdims(self: _MaskedPairOrTensor,
             dim0: _int,
             dim1: _int) -> MaskedPair: ...


def swapdims_(self: _MaskedPairOrTensor,
              dim0: _int,
              dim1: _int) -> MaskedPair: ...


def t(self: _MaskedPairOrTensor) -> MaskedPair: ...


def t_(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def transpose(self: _MaskedPairOrTensor,
              dim0: _int,
              dim1: _int) -> MaskedPair: ...


@overload
def transpose_(self: _MaskedPairOrTensor,
               dim0: _int,
               dim1: _int) -> MaskedPair: ...


@overload
def transpose(self: _MaskedPairOrTensor,
              dim0: _dimname,
              dim1: _dimname) -> MaskedPair: ...


@overload
def transpose_(self: _MaskedPairOrTensor,
               dim0: _dimname,
               dim1: _dimname) -> MaskedPair: ...


def permute(self: _MaskedPairOrTensor,
            dims: Union[_int, _size]) -> MaskedPair: ...


def permute_copy(self: _MaskedPairOrTensor,
                 dims: Union[_int, _size]) -> MaskedPair: ...


def take(self: _MaskedPairOrTensor,
         index: Tensor) -> MaskedPair: ...


def take_along_dim(self: _MaskedPairOrTensor,
                   indices: Tensor,
                   dim: Optional[_int] = None) -> MaskedPair: ...


@overload
def gather(self: _MaskedPairOrTensor,
           dim: _int,
           index: Tensor,
           *,
           sparse_grad: _bool = False) -> MaskedPair: ...


@overload
def gather(self: _MaskedPairOrTensor,
           dim: _dimname,
           index: Tensor,
           *,
           sparse_grad: _bool = False) -> MaskedPair: ...


def unfold(self: _MaskedPairOrTensor,
           dimension: _int,
           size: _int,
           step: _int) -> MaskedPair: ...


def im2col(self: _MaskedPairOrTensor,
           kernel_size: Union[_int, _size],
           stride: Union[_int, _size],
           padding: Union[_int, _size],
           dilation: Union[_int, _size]) -> MaskedPair: ...


def col2im(self: _MaskedPairOrTensor,
           output_size: Union[_symint, Sequence[_symint]],
           kernel_size: Union[_int, _size],
           stride: Union[_int, _size],
           padding: Union[_int, _size],
           dilation: Union[_int, _size]) -> MaskedPair: ...


# one to many
def chunk(self: _MaskedPairOrTensor,
          chunks: _int,
          dim: _int = 0) -> List[MaskedPair]: ...


@overload
def split(self: _MaskedPairOrTensor,
          split_size: _symint,
          dim: _int = 0) -> List[MaskedPair]: ...


@overload
def split(self: _MaskedPairOrTensor,
          split_size: Union[_int, _size],
          dim: _int = 0) -> List[MaskedPair]: ...


def split_with_sizes(self: _MaskedPairOrTensor,
                     split_size: Union[_int, _size],
                     dim: _int = 0) -> List[MaskedPair]: ...


def split_copy(self: _MaskedPairOrTensor,
               split_size: _symint,
               dim: _int = 0) -> List[MaskedPair]: ...


def split_with_sizes_copy(self: _MaskedPairOrTensor,
                          split_size: Union[_int, _size],
                          dim: _int = 0) -> List[MaskedPair]: ...


@overload
def dsplit(self: _MaskedPairOrTensor,
           sections: _int) -> List[MaskedPair]: ...


@overload
def dsplit(self: _MaskedPairOrTensor,
           indices: Union[_int, _size]) -> List[MaskedPair]: ...


@overload
def hsplit(self: _MaskedPairOrTensor,
           sections: _int) -> List[MaskedPair]: ...


@overload
def hsplit(self: _MaskedPairOrTensor,
           indices: Union[_int, _size]) -> List[MaskedPair]: ...


@overload
def vsplit(self: _MaskedPairOrTensor,
           sections: _int) -> List[MaskedPair]: ...


@overload
def vsplit(self: _MaskedPairOrTensor,
           indices: Union[_int, _size]) -> List[MaskedPair]: ...


# many to one
def cat(self: _MaskedPairListOrTensorList,
        dim: _int) -> MaskedPair: ...


def row_stack(self: _MaskedPairListOrTensorList) -> MaskedPair: ...


def column_stack(self: _MaskedPairListOrTensorList) -> MaskedPair: ...


def hstack(self: _MaskedPairListOrTensorList) -> MaskedPair: ...


def vstack(self: _MaskedPairListOrTensorList) -> MaskedPair: ...


# many to many
@overload
def atleast_1d(self: _MaskedPairListOrTensorList) -> List[MaskedPair]: ...


@overload
def atleast_2d(self: _MaskedPairListOrTensorList) -> List[MaskedPair]: ...


@overload
def atleast_3d(self: _MaskedPairListOrTensorList) -> List[MaskedPair]: ...


def broadcast_tensors(self: _MaskedPairListOrTensorList) -> List[MaskedPair]: ...


@overload
def meshgrid(self: _MaskedPairListOrTensorList) -> List[MaskedPair]: ...


@overload
def meshgrid(self: _MaskedPairListOrTensorList,
             *,
             indexing: str) -> List[MaskedPair]: ...
