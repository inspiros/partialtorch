r"""
Partial equivalence of :mod:`~torch.nn.functional`.
"""

from typing import List, Tuple, Optional

import torch.nn.functional
# noinspection PyUnresolvedReferences
from torch._jit_internal import boolean_dispatch, BroadcastingList1, BroadcastingList2, BroadcastingList3

from torch.nn.modules.utils import _pair, _triple

import partialtorch
from partialtorch.types import Tensor, MaskedPair

partial_conv1d = partialtorch.ops.partial_conv1d
partial_conv2d = partialtorch.ops.partial_conv2d
partial_conv3d = partialtorch.ops.partial_conv3d
partial_conv_transpose1d = partialtorch.ops.partial_conv_transpose1d
partial_conv_transpose2d = partialtorch.ops.partial_conv_transpose2d
partial_conv_transpose3d = partialtorch.ops.partial_conv_transpose3d


# TODO: conv_tbc, avg_pool1d, avg_pool2d, avg_pool3d, ...

# noinspection PyUnusedLocal
def fractional_max_pool2d_with_indices(
        input: MaskedPair,
        kernel_size: BroadcastingList2[int],
        output_size: Optional[BroadcastingList2[int]] = None,
        output_ratio: Optional[BroadcastingList2[float]] = None,
        return_indices: bool = False,
        _random_samples: Optional[Tensor] = None) -> Tuple[MaskedPair, Tensor]:
    r"""See :func:`~torch.nn.functional.fractional_max_pool2d` for details.
    """
    if output_size is None and output_ratio is None:
        raise ValueError("fractional_max_pool2d requires specifying either an output_size or an output_ratio")
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _pair(output_ratio)
        output_size = [int(input.size(-2) * _output_ratio[0]), int(input.size(-1) * _output_ratio[1])]

    if _random_samples is None:
        n_batch = 1 if input.dim() == 3 else input.size(0)
        _random_samples = torch.rand(n_batch, input.size(-3), 2, dtype=input.dtype, device=input.device)
    return partialtorch.ops.fractional_max_pool2d(input, kernel_size, output_size, _random_samples)


def _fractional_max_pool2d(
        input: MaskedPair,
        kernel_size: BroadcastingList2[int],
        output_size: Optional[BroadcastingList2[int]] = None,
        output_ratio: Optional[BroadcastingList2[float]] = None,
        return_indices: bool = False,
        _random_samples: Optional[Tensor] = None) -> MaskedPair:
    return fractional_max_pool2d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool2d_with_indices,
    if_false=_fractional_max_pool2d,
    module_name=__name__,
    func_name="fractional_max_pool2d",
)

# noinspection PyUnusedLocal
def fractional_max_pool3d_with_indices(
        input: MaskedPair,
        kernel_size: BroadcastingList3[int],
        output_size: Optional[BroadcastingList3[int]] = None,
        output_ratio: Optional[BroadcastingList3[float]] = None,
        return_indices: bool = False,
        _random_samples: Optional[Tensor] = None) -> Tuple[MaskedPair, Tensor]:
    r"""See :func:`~torch.nn.functional.fractional_max_pool3d` for details.
    """
    if output_size is None and output_ratio is None:
        raise ValueError("fractional_max_pool3d requires specifying either an output_size or an output_ratio")
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _triple(output_ratio)
        output_size = [
            int(input.size(-3) * _output_ratio[0]),
            int(input.size(-2) * _output_ratio[1]),
            int(input.size(-1) * _output_ratio[2]),
        ]

    if _random_samples is None:
        n_batch = 1 if input.dim() == 4 else input.size(0)
        _random_samples = torch.rand(n_batch, input.size(-4), 3, dtype=input.dtype, device=input.device)
    return partialtorch.ops.fractional_max_pool3d(input, kernel_size, output_size, _random_samples)


def _fractional_max_pool3d(
        input: MaskedPair,
        kernel_size: BroadcastingList3[int],
        output_size: Optional[BroadcastingList3[int]] = None,
        output_ratio: Optional[BroadcastingList3[float]] = None,
        return_indices: bool = False,
        _random_samples: Optional[Tensor] = None) -> MaskedPair:
    return fractional_max_pool3d_with_indices(
        input, kernel_size, output_size, output_ratio, return_indices, _random_samples
    )[0]


fractional_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=4,
    default=False,
    if_true=fractional_max_pool3d_with_indices,
    if_false=_fractional_max_pool3d,
    module_name=__name__,
    func_name="fractional_max_pool3d",
)


# noinspection PyUnusedLocal
def max_pool1d_with_indices(
        input: MaskedPair,
        kernel_size: BroadcastingList1[int],
        stride: Optional[BroadcastingList1[int]] = None,
        padding: BroadcastingList1[int] = 0,
        dilation: BroadcastingList1[int] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False) -> Tuple[MaskedPair, Tensor]:
    r"""See :func:`~torch.nn.functional.max_pool1d` for details.
    """
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return partialtorch.ops.max_pool1d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)


# noinspection PyUnusedLocal
def _max_pool1d(
        input: MaskedPair,
        kernel_size: BroadcastingList1[int],
        stride: Optional[BroadcastingList1[int]] = None,
        padding: BroadcastingList1[int] = 0,
        dilation: BroadcastingList1[int] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False) -> MaskedPair:
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return partialtorch.ops.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool1d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool1d_with_indices,
    if_false=_max_pool1d,
    module_name=__name__,
    func_name="max_pool1d",
)


# noinspection PyUnusedLocal
def max_pool2d_with_indices(
        input: MaskedPair,
        kernel_size: BroadcastingList2[int],
        stride: Optional[BroadcastingList2[int]] = None,
        padding: BroadcastingList2[int] = 0,
        dilation: BroadcastingList2[int] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False) -> Tuple[MaskedPair, Tensor]:
    r"""See :func:`~torch.nn.functional.max_pool2d` for details.
    """
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return partialtorch.ops.max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)


# noinspection PyUnusedLocal
def _max_pool2d(
        input: MaskedPair,
        kernel_size: BroadcastingList2[int],
        stride: Optional[BroadcastingList2[int]] = None,
        padding: BroadcastingList2[int] = 0,
        dilation: BroadcastingList2[int] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False) -> MaskedPair:
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return partialtorch.ops.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool2d_with_indices,
    if_false=_max_pool2d,
    module_name=__name__,
    func_name="max_pool2d",
)


# noinspection PyUnusedLocal
def max_pool3d_with_indices(
        input: MaskedPair,
        kernel_size: BroadcastingList3[int],
        stride: Optional[BroadcastingList3[int]] = None,
        padding: BroadcastingList3[int] = 0,
        dilation: BroadcastingList3[int] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False) -> Tuple[MaskedPair, Tensor]:
    r"""See :func:`~torch.nn.functional.max_pool3d` for details.
    """
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return partialtorch.ops.max_pool3d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)


# noinspection PyUnusedLocal
def _max_pool3d(
        input: MaskedPair,
        kernel_size: BroadcastingList3[int],
        stride: Optional[BroadcastingList3[int]] = None,
        padding: BroadcastingList3[int] = 0,
        dilation: BroadcastingList3[int] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False) -> MaskedPair:
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return partialtorch.ops.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)


max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=6,
    default=False,
    if_true=max_pool3d_with_indices,
    if_false=_max_pool3d,
    module_name=__name__,
    func_name="max_pool3d",
)


# TODO: max_unpool, lp_pool, adaptive_max_pool, adaptive_avg_pool


def dropout(input: MaskedPair, p: float = 0.5, training: bool = True, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.dropout` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.dropout_(input, p, training) if inplace \
        else partialtorch.ops.dropout(input, p, training)


def alpha_dropout(input: MaskedPair, p: float = 0.5, training: bool = False, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.alpha_dropout` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.alpha_dropout_(input, p, training) if inplace \
        else partialtorch.ops.alpha_dropout(input, p, training)


def dropout1d(input: MaskedPair, p: float = 0.5, training: bool = True, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.dropout1d` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.dropout1d_(input, p, training) if inplace \
        else partialtorch.ops.dropout1d(input, p, training)


def dropout2d(input: MaskedPair, p: float = 0.5, training: bool = True, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.dropout2d` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.dropout2d_(input, p, training) if inplace \
        else partialtorch.ops.dropout2d(input, p, training)


def dropout3d(input: MaskedPair, p: float = 0.5, training: bool = True, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.dropout3d` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.dropout3d_(input, p, training) if inplace \
        else partialtorch.ops.dropout3d(input, p, training)


def feature_alpha_dropout(input: MaskedPair, p: float = 0.5, training: bool = False,
                          inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.feature_alpha_dropout` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.feature_alpha_dropout_(input, p, training) if inplace \
        else partialtorch.ops.feature_alpha_dropout(input, p, training)


def _threshold(input: MaskedPair, threshold: float, value: float, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.threshold` for details.
    """
    return partialtorch.ops.threshold_(input, threshold, value) if inplace \
        else partialtorch.ops.threshold(input, threshold, value)


# We define this function as _threshold because it takes an argument named threshold
threshold = _threshold
threshold_ = partialtorch.ops.threshold_


def relu(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.relu` for details.
    """
    return partialtorch.ops.relu_(input) if inplace \
        else partialtorch.ops.relu(input)


relu_ = partialtorch.ops.relu_


def glu(input: MaskedPair, dim: int = -1) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.glu` for details.
    """
    if input.dim() == 0:
        raise RuntimeError("glu does not support scalars because halving size must be even")
    return partialtorch.ops.glu(input, dim)


def hardtanh(input: MaskedPair, min_val: float = -1., max_val: float = 1., inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.hardtanh` for details.
    """
    return partialtorch.ops.hardtanh_(input, min_val, max_val) if inplace \
        else partialtorch.ops.hardtanh(input, min_val, max_val)


hardtanh_ = partialtorch.ops.hardtanh_


def relu6(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.relu6` for details.
    """
    return partialtorch.ops.relu6_(input) if inplace \
        else partialtorch.ops.relu6(input)


def elu(input: MaskedPair, alpha: float = 1.0, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.elu` for details.
    """
    return partialtorch.ops.elu_(input, alpha) if inplace \
        else partialtorch.ops.elu(input, alpha)


elu_ = partialtorch.ops.elu_


def selu(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.selu` for details.
    """
    return partialtorch.ops.selu_(input) if inplace \
        else partialtorch.ops.selu(input)


selu_ = partialtorch.ops.selu_


def celu(input: MaskedPair, alpha: float = 1.0, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.celu` for details.
    """
    return partialtorch.ops.celu_(input, alpha) if inplace \
        else partialtorch.ops.celu(input, alpha)


celu_ = partialtorch.ops.celu_


def leaky_relu(input: MaskedPair, negative_slope: float = 0.01, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.leaky_relu` for details.
    """
    return partialtorch.ops.leaky_relu_(input, negative_slope) if inplace \
        else partialtorch.ops.leaky_relu(input, negative_slope)


leaky_relu_ = partialtorch.ops.leaky_relu_

prelu = partialtorch.ops.prelu


def rrelu(input: MaskedPair, lower: float = 1.0 / 8, upper: float = 1.0 / 3, training: bool = False,
          inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.rrelu` for details.
    """
    return partialtorch.ops.rrelu_(input, lower, upper, training) if inplace \
        else partialtorch.ops.rrelu(input, lower, upper, training)


rrelu_ = partialtorch.ops.rrelu_
logsigmoid = partialtorch.ops.logsigmoid
gelu = partialtorch.ops.gelu
hardshrink = partialtorch.ops.hardshrink
tanhshrink = partialtorch.ops.tanhshrink
softsign = partialtorch.ops.softsign
softplus = partialtorch.ops.softplus
softmin = partialtorch.ops.softmin
softmax = partialtorch.ops.softmax
# TODO: gumbel_softmax
log_softmax = partialtorch.ops.log_softmax
softshrink = partialtorch.ops.softshrink
tanh = partialtorch.ops.tanh
sigmoid = partialtorch.ops.sigmoid


def hardsigmoid(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.hardsigmoid` for details.
    """
    return partialtorch.ops.hardsigmoid_(input) if inplace \
        else partialtorch.ops.hardsigmoid(input)


partial_linear = partialtorch.ops.partial_linear
partial_bilinear = partialtorch.ops.partial_bilinear


def silu(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.silu` for details.
    """
    return partialtorch.ops.silu_(input) if inplace \
        else partialtorch.ops.silu(input)


def mish(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.mish` for details.
    """
    return partialtorch.ops.mish_(input) if inplace \
        else partialtorch.ops.mish(input)


def hardswish(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.hardswish` for details.
    """
    return partialtorch.ops.hardswish_(input) if inplace \
        else partialtorch.ops.hardswish(input)


# TODO: embedding, embedding_bag


def batch_norm(
        input: MaskedPair,
        running_mean: Optional[Tensor],
        running_var: Optional[Tensor],
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5) -> MaskedPair:
    r"""See :func:`~torch.nn.functional.batch_norm` for details.
    """
    return partialtorch.ops.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps)

# TODO: instance_norm, layer_norm, group_norm, local_response_norm
