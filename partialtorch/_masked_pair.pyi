from typing import overload, Tuple, Optional, Union

import torch

from partialtorch.types import (
    _int, _bool, _dtype, _device, _size, _layout, _memory_format, Number, Storage, Tensor, _TensorOrTensors
)


class MaskedPair:
    data: Tensor
    mask: Optional[Tensor]

    def __init__(self, data: Tensor, mask: Optional[Tensor]): ...

    members: Tuple[Tensor, Optional[Tensor]] = property(lambda self: object(), lambda self, v: None)
    shape: Tuple[_int, ...] = property(lambda self: object())
    ndim: _int = property(lambda self: object())
    layout: _layout = property(lambda self: object())
    dtype: _dtype = property(lambda self: object())
    device: _device = property(lambda self: object())
    is_cpu: _bool = property(lambda self: object())
    is_cuda: _bool = property(lambda self: object())
    is_ipu: _bool = property(lambda self: object())
    is_xpu: _bool = property(lambda self: object())
    is_sparse: _bool = property(lambda self: object())
    is_sparse_csr: _bool = property(lambda self: object())
    is_mkldnn: _bool = property(lambda self: object())
    is_mps: _bool = property(lambda self: object())
    is_ort: _bool = property(lambda self: object())
    is_vulkan: _bool = property(lambda self: object())
    is_quantized: _bool = property(lambda self: object())
    is_meta: _bool = property(lambda self: object())
    is_nested: _bool = property(lambda self: object())
    requires_grad: _bool = property(lambda self: object())
    is_leaf: _bool = property(lambda self: object())
    output_nr: _int = property(lambda self: object())
    _version: _int = property(lambda self: object())
    retains_grad: _bool = property(lambda self: object())
    grad: Tensor = property(lambda self: object(), lambda self, v: None)

    def dim(self) -> _int: ...

    def storage_offset(self) -> _int: ...

    def is_complex(self) -> _bool: ...

    def is_floating_point(self) -> _bool: ...

    def is_signed(self) -> _bool: ...

    def size(self, dim: _int) -> _int: ...

    def stride(self) -> Tuple[_int, ...]: ...

    def sizes(self) -> Tuple[_int, ...]: ...

    def ndimension(self) -> _int: ...

    def is_contiguous(self, memory_format: _memory_format = torch.contiguous_format) -> _bool: ...

    def numel(self) -> _int: ...

    def element_size(self) -> _int: ...

    def storage(self) -> Storage: ...

    def is_conj(self) -> _bool: ...

    def is_neg(self) -> _bool: ...

    def get_device(self) -> _int: ...

    def is_inference(self) -> _bool: ...

    def has_names(self) -> _bool: ...

    def retain_grad(self) -> None: ...

    def requires_grad_(self, _requires_grad: _bool = True) -> MaskedPair: ...

    def cpu(self) -> MaskedPair: ...

    def cuda(self,
             device: _device = None,
             non_blocking: _bool = False,
             memory_format: _memory_format = torch.preserve_format) -> MaskedPair: ...

    def backward(self,
                 gradient: Tensor = None,
                 retain_graph: Optional[_bool] = None,
                 create_graph: _bool = False,
                 inputs: Optional[_TensorOrTensors] = None) -> None: ...

    def clone(self, memory_format: Optional[_memory_format] = None) -> MaskedPair: ...

    def contiguous(self, *, memory_format: _memory_format = torch.contiguous_format) -> MaskedPair: ...

    def detach(self) -> MaskedPair: ...

    def detach_(self) -> MaskedPair: ...

    def fill_masked(self, value: Number) -> Tensor: ...

    def fill_masked_(self, value: Number) -> Tensor: ...

    def to(self, *,
           dtype: Optional[_dtype] = None,
           layout: Optional[_layout] = None,
           device: Optional[_device] = None,
           pin_memory: Optional[bool] = None,
           non_blocking: bool = False,
           copy: bool = False,
           memory_format: Optional[_memory_format] = None) -> MaskedPair: ...

    def to_tensor(self, value: Number) -> Tensor: ...

    def item(self, value: Number) -> Number: ...

    def index_non_masked(self) -> Tensor: ...

    def view(self, size: _size) -> MaskedPair: ...

    def t(self) -> MaskedPair: ...

    def t_(self) -> MaskedPair: ...

    def transpose(self) -> MaskedPair: ...

    def transpose_(self) -> MaskedPair: ...

    def permute(self, dims: _size) -> MaskedPair: ...


# creation ops
@overload
def masked_pair(data: Tensor, mask: Optional[Tensor] = None) -> MaskedPair: ...


@overload
def masked_pair(data: Tensor, mask: Optional[_bool]) -> MaskedPair: ...


@overload
def masked_pair(args: Union[Tuple[Tensor], Tuple[Tensor, Tensor]]) -> MaskedPair: ...


@overload
def masked_pair(input: MaskedPair) -> MaskedPair: ...
