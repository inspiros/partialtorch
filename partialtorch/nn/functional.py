r"""
Masked equivalence of :mod:`~torch.nn.functional`.
"""
import math
import warnings
from typing import List, Tuple, Optional, Union

import torch.nn.functional
# noinspection PyUnresolvedReferences
from torch._jit_internal import (
    _overload, boolean_dispatch, BroadcastingList1, BroadcastingList2, BroadcastingList3
)
from torch.nn.modules.utils import _pair, _triple

import partialtorch
from partialtorch.types import Tensor, MaskedPair

partial_conv1d = partialtorch.ops.partial_conv1d
partial_conv2d = partialtorch.ops.partial_conv2d
partial_conv3d = partialtorch.ops.partial_conv3d
partial_conv_transpose1d = partialtorch.ops.partial_conv_transpose1d
partial_conv_transpose2d = partialtorch.ops.partial_conv_transpose2d
partial_conv_transpose3d = partialtorch.ops.partial_conv_transpose3d
partial_conv_tbc = partialtorch.ops.partial_conv_tbc
avg_pool1d = partialtorch.ops.avg_pool1d
avg_pool2d = partialtorch.ops.avg_pool2d
avg_pool3d = partialtorch.ops.avg_pool3d


# noinspection PyUnusedLocal
def fractional_max_pool2d_with_indices(
        input: MaskedPair,
        kernel_size: BroadcastingList2[int],
        output_size: Optional[BroadcastingList2[int]] = None,
        output_ratio: Optional[BroadcastingList2[float]] = None,
        return_indices: bool = False,
        _random_samples: Optional[Tensor] = None) -> Tuple[MaskedPair, Tensor]:
    r"""See :func:`torch.nn.functional.fractional_max_pool2d` for details.
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
    r"""See :func:`torch.nn.functional.fractional_max_pool3d` for details.
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
    r"""See :func:`torch.nn.functional.max_pool1d` for details.
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
    r"""See :func:`torch.nn.functional.max_pool2d` for details.
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
    r"""See :func:`torch.nn.functional.max_pool3d` for details.
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


# TODO: max_unpool

def lp_pool1d(
        input: MaskedPair,
        norm_type: Union[int, float],
        kernel_size: int,
        stride: Optional[BroadcastingList1[int]] = None,
        ceil_mode: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.lp_pool1d` for details.
    """
    if stride is not None:
        out = avg_pool1d(partialtorch.pow(input, norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool1d(partialtorch.pow(input, norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return partialtorch.pow(partialtorch.mul(partialtorch.mul(
        partialtorch.sign(out), relu(partialtorch.abs(out))), kernel_size), 1.0 / norm_type)


def lp_pool2d(
        input: MaskedPair,
        norm_type: Union[int, float],
        kernel_size: BroadcastingList2[int],
        stride: Optional[BroadcastingList2[int]] = None,
        ceil_mode: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.lp_pool2d` for details.
    """
    kw, kh = _pair(kernel_size)
    if stride is not None:
        out = avg_pool2d(partialtorch.pow(input, norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool2d(partialtorch.pow(input, norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)

    return partialtorch.pow(partialtorch.mul(partialtorch.mul(
        partialtorch.sign(out), relu(partialtorch.abs(out))), kw * kh), 1.0 / norm_type)


def adaptive_max_pool1d_with_indices(
        input: MaskedPair,
        output_size: BroadcastingList1[int],
        return_indices: bool = False) -> Tuple[MaskedPair, Tensor]:
    r"""See :func:`torch.nn.functional.adaptive_max_pool1d_with_indices` for details.
    """
    return partialtorch.ops.adaptive_max_pool1d(input, output_size)


def _adaptive_max_pool1d(
        input: MaskedPair,
        output_size: BroadcastingList1[int],
        return_indices: bool = False) -> MaskedPair:
    return adaptive_max_pool1d_with_indices(input, output_size)[0]


adaptive_max_pool1d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool1d_with_indices,
    if_false=_adaptive_max_pool1d,
    module_name=__name__,
    func_name="adaptive_max_pool1d",
)


def adaptive_max_pool2d_with_indices(
        input: MaskedPair,
        output_size: BroadcastingList1[int],
        return_indices: bool = False) -> Tuple[MaskedPair, Tensor]:
    r"""See :func:`torch.nn.functional.adaptive_max_pool2d_with_indices` for details.
    """
    return partialtorch.ops.adaptive_max_pool2d(input, output_size)


def _adaptive_max_pool2d(
        input: MaskedPair,
        output_size: BroadcastingList1[int],
        return_indices: bool = False) -> MaskedPair:
    return adaptive_max_pool2d_with_indices(input, output_size)[0]


adaptive_max_pool2d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool2d_with_indices,
    if_false=_adaptive_max_pool2d,
    module_name=__name__,
    func_name="adaptive_max_pool2d",
)


def adaptive_max_pool3d_with_indices(
        input: MaskedPair,
        output_size: BroadcastingList1[int],
        return_indices: bool = False) -> Tuple[MaskedPair, Tensor]:
    r"""See :func:`torch.nn.functional.adaptive_max_pool3d_with_indices` for details.
    """
    return partialtorch.ops.adaptive_max_pool3d(input, output_size)


def _adaptive_max_pool3d(
        input: MaskedPair,
        output_size: BroadcastingList1[int],
        return_indices: bool = False) -> MaskedPair:
    return adaptive_max_pool3d_with_indices(input, output_size)[0]


adaptive_max_pool3d = boolean_dispatch(
    arg_name="return_indices",
    arg_index=2,
    default=False,
    if_true=adaptive_max_pool3d_with_indices,
    if_false=_adaptive_max_pool3d,
    module_name=__name__,
    func_name="adaptive_max_pool3d",
)

adaptive_avg_pool1d = partialtorch.ops.adaptive_avg_pool1d
adaptive_avg_pool2d = partialtorch.ops.adaptive_avg_pool2d
adaptive_avg_pool3d = partialtorch.ops.adaptive_avg_pool3d


def dropout(input: MaskedPair, p: float = 0.5, training: bool = True, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.dropout` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.dropout_(input, p, training) if inplace \
        else partialtorch.ops.dropout(input, p, training)


def alpha_dropout(input: MaskedPair, p: float = 0.5, training: bool = False, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.alpha_dropout` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.alpha_dropout_(input, p, training) if inplace \
        else partialtorch.ops.alpha_dropout(input, p, training)


def dropout1d(input: MaskedPair, p: float = 0.5, training: bool = True, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.dropout1d` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.dropout1d_(input, p, training) if inplace \
        else partialtorch.ops.dropout1d(input, p, training)


def dropout2d(input: MaskedPair, p: float = 0.5, training: bool = True, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.dropout2d` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.dropout2d_(input, p, training) if inplace \
        else partialtorch.ops.dropout2d(input, p, training)


def dropout3d(input: MaskedPair, p: float = 0.5, training: bool = True, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.dropout3d` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.dropout3d_(input, p, training) if inplace \
        else partialtorch.ops.dropout3d(input, p, training)


def feature_alpha_dropout(input: MaskedPair, p: float = 0.5, training: bool = False,
                          inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.feature_alpha_dropout` for details.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
    return partialtorch.ops.feature_alpha_dropout_(input, p, training) if inplace \
        else partialtorch.ops.feature_alpha_dropout(input, p, training)


def _threshold(input: MaskedPair, threshold: float, value: float, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.threshold` for details.
    """
    return partialtorch.ops.threshold_(input, threshold, value) if inplace \
        else partialtorch.ops.threshold(input, threshold, value)


# We define this function as _threshold because it takes an argument named threshold
threshold = _threshold
threshold_ = partialtorch.ops.threshold_


def relu(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.relu` for details.
    """
    return partialtorch.ops.relu_(input) if inplace \
        else partialtorch.ops.relu(input)


relu_ = partialtorch.ops.relu_


def glu(input: MaskedPair, dim: int = -1) -> MaskedPair:
    r"""See :func:`torch.nn.functional.glu` for details.
    """
    if input.dim() == 0:
        raise RuntimeError("glu does not support scalars because halving size must be even")
    return partialtorch.ops.glu(input, dim)


def hardtanh(input: MaskedPair, min_val: float = -1., max_val: float = 1., inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.hardtanh` for details.
    """
    return partialtorch.ops.hardtanh_(input, min_val, max_val) if inplace \
        else partialtorch.ops.hardtanh(input, min_val, max_val)


hardtanh_ = partialtorch.ops.hardtanh_


def relu6(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.relu6` for details.
    """
    return partialtorch.ops.relu6_(input) if inplace \
        else partialtorch.ops.relu6(input)


def elu(input: MaskedPair, alpha: float = 1.0, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.elu` for details.
    """
    return partialtorch.ops.elu_(input, alpha) if inplace \
        else partialtorch.ops.elu(input, alpha)


elu_ = partialtorch.ops.elu_


def selu(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.selu` for details.
    """
    return partialtorch.ops.selu_(input) if inplace \
        else partialtorch.ops.selu(input)


selu_ = partialtorch.ops.selu_


def celu(input: MaskedPair, alpha: float = 1.0, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.celu` for details.
    """
    return partialtorch.ops.celu_(input, alpha) if inplace \
        else partialtorch.ops.celu(input, alpha)


celu_ = partialtorch.ops.celu_


def leaky_relu(input: MaskedPair, negative_slope: float = 0.01, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.leaky_relu` for details.
    """
    return partialtorch.ops.leaky_relu_(input, negative_slope) if inplace \
        else partialtorch.ops.leaky_relu(input, negative_slope)


leaky_relu_ = partialtorch.ops.leaky_relu_

prelu = partialtorch.ops.prelu


def rrelu(input: MaskedPair, lower: float = 1.0 / 8, upper: float = 1.0 / 3, training: bool = False,
          inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.rrelu` for details.
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


def gumbel_softmax(logits: MaskedPair,
                   tau: float = 1,
                   hard: bool = False,
                   eps: float = 1e-10,
                   dim: int = -1) -> MaskedPair:
    r"""See :func:`torch.nn.functional.gumbel_softmax` for details.
    """
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    logits = partialtorch.masked_pair(logits)
    gumbels = (
        -torch.empty_like(logits.data,
                          memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = partialtorch.div(partialtorch.add(logits, gumbels), tau)  # ~Gumbel(logits,tau)
    y_soft = partialtorch.softmax(gumbels, dim)

    if hard:
        # Straight through.
        index = partialtorch.max(y_soft, dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits.data,
                                  memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = partialtorch.add(partialtorch.sub(y_hard, y_soft.detach()), y_soft)
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


log_softmax = partialtorch.ops.log_softmax
softshrink = partialtorch.ops.softshrink
tanh = partialtorch.ops.tanh
sigmoid = partialtorch.ops.sigmoid


def hardsigmoid(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.hardsigmoid` for details.
    """
    return partialtorch.ops.hardsigmoid_(input) if inplace \
        else partialtorch.ops.hardsigmoid(input)


partial_linear = partialtorch.ops.partial_linear
partial_bilinear = partialtorch.ops.partial_bilinear


def silu(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.silu` for details.
    """
    return partialtorch.ops.silu_(input) if inplace \
        else partialtorch.ops.silu(input)


def mish(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.mish` for details.
    """
    return partialtorch.ops.mish_(input) if inplace \
        else partialtorch.ops.mish(input)


def hardswish(input: MaskedPair, inplace: bool = False) -> MaskedPair:
    r"""See :func:`torch.nn.functional.hardswish` for details.
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
    r"""See :func:`torch.nn.functional.batch_norm` for details.
    """
    if training:
        torch.nn.functional._verify_batch_size(input.shape)
    return partialtorch.ops.batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps, torch.backends.cudnn.enabled)


def instance_norm(
        input: MaskedPair,
        running_mean: Optional[Tensor] = None,
        running_var: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        use_input_stats: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5) -> MaskedPair:
    r"""See :func:`torch.nn.functional.instance_norm` for details.
    """
    if use_input_stats:
        torch.nn.functional._verify_spatial_size(input.shape)
    return partialtorch.ops.instance_norm(
        input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, torch.backends.cudnn.enabled)


def layer_norm(
        input: MaskedPair,
        normalized_shape: List[int],
        weight: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5) -> MaskedPair:
    r"""See :func:`torch.nn.functional.layer_norm` for details.
    """
    return partialtorch.ops.layer_norm(
        input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)


# TODO: group_norm, local_response_norm

# loss
# TODO: ctc_loss, nll_loss, poisson_nll_loss, gaussian_nll_loss, kl_div, cross_entropy,
#  binary_cross_entropy, binary_cross_entropy_with_logits, smooth_l1_loss, huber_loss, l1_loss,
#  mse_loss, margin_ranking_loss, hinge_embedding_loss, multilabel_margin_loss, soft_margin_loss,
#  multilabel_soft_margin_loss, cosine_embedding_loss, multi_margin_loss

pixel_shuffle = partialtorch.ops.pixel_shuffle
pixel_unshuffle = partialtorch.ops.pixel_unshuffle
channel_shuffle = partialtorch.ops.channel_shuffle
native_channel_shuffle = partialtorch.ops.native_channel_shuffle


@_overload  # noqa: F811
def upsample(input: MaskedPair,
             size: Optional[int] = None,
             scale_factor: Optional[float] = None,
             mode: str = "nearest",
             align_corners: Optional[bool] = None) -> MaskedPair:  # noqa: F811,B950
    pass


@_overload  # noqa: F811
def upsample(input: MaskedPair,
             size: Optional[List[int]] = None,
             scale_factor: Optional[float] = None,
             mode: str = "nearest",
             align_corners: Optional[bool] = None) -> MaskedPair:  # noqa: F811,B950
    pass


def upsample(input, size=None, scale_factor=None, mode="nearest", align_corners=None):  # noqa: F811
    r"""See :func:`torch.nn.functional.upsample` for details.

    .. warning::
        This function is deprecated in favor of :func:`partialtorch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(...)``.
    """
    warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
    return interpolate(input, size, scale_factor, mode, align_corners)


@_overload  # noqa: F811
def interpolate(input: MaskedPair,
                size: Optional[int] = None, scale_factor: Optional[List[float]] = None,
                mode: str = 'nearest',
                align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None,
                antialias: bool = False) -> MaskedPair:  # noqa: F811,B950
    pass


@_overload  # noqa: F811
def interpolate(input: MaskedPair,
                size: Optional[List[int]] = None,
                scale_factor: Optional[List[float]] = None,
                mode: str = 'nearest',
                align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None,
                antialias: bool = False) -> MaskedPair:  # noqa: F811,B950
    pass


@_overload  # noqa: F811
def interpolate(input: MaskedPair,
                size: Optional[int] = None,
                scale_factor: Optional[float] = None,
                mode: str = 'nearest',
                align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None,
                antialias: bool = False) -> MaskedPair:  # noqa: F811,B950
    pass


@_overload  # noqa: F811
def interpolate(input: MaskedPair,
                size: Optional[List[int]] = None,
                scale_factor: Optional[float] = None,
                mode: str = "nearest",
                align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None,
                antialias: bool = False) -> MaskedPair:  # noqa: F811
    pass


def interpolate(input: MaskedPair,
                size: Optional[int] = None,
                scale_factor: Optional[List[float]] = None,
                mode: str = 'nearest',
                align_corners: Optional[bool] = None,
                recompute_scale_factor: Optional[bool] = None,
                antialias: bool = False) -> MaskedPair:  # noqa: F811,B950
    r"""See :func:`torch.nn.functional.interpolate` for details.
    """
    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        if align_corners is None:
            align_corners = False

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "Input and output must have the same number of spatial dimensions, but got "
                    f"input with spatial dimensions of {list(input.shape[2:])} and output size of {size}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "output size in (o1, o2, ...,oK) format."
                )
            if not torch.jit.is_scripting():
                if not all(torch.nn.functional._is_integer(x) for x in size):
                    raise TypeError(
                        "expected size to be one of int or Tuple[int] or Tuple[int, int] or "
                        f"Tuple[int, int, int], but got size with types {[type(x) for x in size]}"
                    )
            output_size = size
        else:
            output_size = [size for _ in range(dim)]
    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "Input and scale_factor must have the same number of spatial dimensions, but "
                    f"got input with spatial dimensions of {list(input.shape[2:])} and "
                    f"scale_factor of shape {scale_factor}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "scale_factor in (s1, s2, ...,sK) format."
                )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
    else:
        raise ValueError("either size or scale_factor should be defined")

    if recompute_scale_factor is not None and recompute_scale_factor and size is not None:
        raise ValueError("recompute_scale_factor is not meaningful with an explicit size.")

    # "area" mode always requires an explicit size rather than scale factor.
    # Re-use the recompute_scale_factor code path.
    if mode == "area" and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        # We compute output_size here, then un-set scale_factors.
        # The C++ code will recompute it based on the (integer) output size.
        assert scale_factors is not None
        if not torch.jit.is_scripting() and torch._C._get_tracing_state():
            # make scale_factor a tensor in tracing so constant doesn't get baked in
            output_size = [
                (torch.floor((input.size(i + 2).float() * torch.tensor(scale_factors[i], dtype=torch.float32)).float()))
                for i in range(dim)
            ]
        elif torch.jit.is_scripting():
            output_size = [int(math.floor(float(input.size(i + 2)) * scale_factors[i]))
                           for i in range(dim)]
        else:
            output_size = [
                torch.sym_int(input.size(i + 2) * scale_factors[i])
                for i in range(dim)
            ]
        scale_factors = None

    if antialias and not (mode in ("bilinear", "bicubic") and input.ndim == 4):
        raise ValueError("Anti-alias option is only supported for bilinear and bicubic modes")

    if input.dim() == 3 and mode == "nearest":
        return partialtorch.ops.upsample_nearest1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest":
        return partialtorch.ops.upsample_nearest2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest":
        return partialtorch.ops.upsample_nearest3d(input, output_size, scale_factors)

    if input.dim() == 3 and mode == "nearest-exact":
        return partialtorch.ops._upsample_nearest_exact1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest-exact":
        return partialtorch.ops._upsample_nearest_exact2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest-exact":
        return partialtorch.ops._upsample_nearest_exact3d(input, output_size, scale_factors)

    if input.dim() == 3 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool1d(input, output_size)
    if input.dim() == 4 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool2d(input, output_size)
    if input.dim() == 5 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool3d(input, output_size)

    if input.dim() == 3 and mode == "linear":
        assert align_corners is not None
        return partialtorch.ops.upsample_linear1d(input, output_size, align_corners, scale_factors)
    if input.dim() == 4 and mode == "bilinear":
        assert align_corners is not None
        if antialias:
            return partialtorch.ops._upsample_bilinear2d_aa(input, output_size, align_corners, scale_factors)
        # No implementation for torch._decomp.decompositions.upsample_bilinear2d_vec
        return partialtorch.ops.upsample_bilinear2d(input, output_size, align_corners, scale_factors)
    if input.dim() == 5 and mode == "trilinear":
        assert align_corners is not None
        return partialtorch.ops.upsample_trilinear3d(input, output_size, align_corners, scale_factors)
    if input.dim() == 4 and mode == "bicubic":
        assert align_corners is not None
        if antialias:
            return partialtorch.ops._upsample_bicubic2d_aa(input, output_size, align_corners, scale_factors)
        return partialtorch.ops.upsample_bicubic2d(input, output_size, align_corners, scale_factors)

    if input.dim() == 3 and mode == "bilinear":
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    if input.dim() == 3 and mode == "trilinear":
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    if input.dim() == 4 and mode == "linear":
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    if input.dim() == 4 and mode == "trilinear":
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    if input.dim() == 5 and mode == "linear":
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    if input.dim() == 5 and mode == "bilinear":
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")

    raise NotImplementedError(
        "Input Error: Only 3D, 4D and 5D input Tensors supported"
        f" (got {input.dim()}D) for the modes: nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact"
        f" (got {mode})"
    )


@_overload  # noqa: F811
def upsample_nearest(input: MaskedPair,
                     size: Optional[int] = None,
                     scale_factor: Optional[float] = None) -> MaskedPair:  # noqa: F811
    pass


@_overload  # noqa: F811
def upsample_nearest(input: MaskedPair,
                     size: Optional[List[int]] = None,
                     scale_factor: Optional[float] = None) -> MaskedPair:  # noqa: F811
    pass


def upsample_nearest(input, size=None, scale_factor=None):  # noqa: F811
    r"""See :func:`torch.nn.functional.upsample_nearest` for details.

    .. warning::
        This function is deprecated in favor of :func:`partialtorch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(..., mode='nearest')``.
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.upsample_nearest is deprecated. Use nn.functional.interpolate instead.")
    return interpolate(input, size, scale_factor, mode="nearest")


@_overload  # noqa: F811
def upsample_bilinear(input: MaskedPair,
                      size: Optional[int] = None,
                      scale_factor: Optional[float] = None) -> MaskedPair:  # noqa: F811
    pass


@_overload  # noqa: F811
def upsample_bilinear(input: MaskedPair,
                      size: Optional[List[int]] = None,
                      scale_factor: Optional[float] = None) -> MaskedPair:  # noqa: F811
    pass


@_overload  # noqa: F811
def upsample_bilinear(input: MaskedPair,
                      size: Optional[int] = None,
                      scale_factor: Optional[List[float]] = None) -> MaskedPair:  # noqa: F811
    pass


@_overload  # noqa: F811
def upsample_bilinear(input: MaskedPair,
                      size: Optional[List[int]] = None,
                      scale_factor: Optional[List[float]] = None) -> MaskedPair:  # noqa: F811
    pass


def upsample_bilinear(input, size=None, scale_factor=None):  # noqa: F811
    r"""See :func:`torch.nn.functional.upsample_bilinear` for details.

    .. warning::
        This function is deprecated in favor of :func:`partialtorch.nn.functional.interpolate`.
        This is equivalent with
        ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead.")
    return interpolate(input, size, scale_factor, mode="bilinear", align_corners=True)


@_overload  # noqa: F811
def partial_upsample(input: MaskedPair,
                     size: Optional[int] = None,
                     scale_factor: Optional[float] = None,
                     mode: str = "nearest",
                     align_corners: Optional[bool] = None) -> MaskedPair:  # noqa: F811,B950
    pass


@_overload  # noqa: F811
def partial_upsample(input: MaskedPair,
                     size: Optional[List[int]] = None,
                     scale_factor: Optional[float] = None,
                     mode: str = "nearest",
                     align_corners: Optional[bool] = None) -> MaskedPair:  # noqa: F811,B950
    pass


def partial_upsample(input, size=None, scale_factor=None, mode="nearest", align_corners=None):  # noqa: F811
    r"""Partial variant of :func:`torch.nn.functional.upsample`.

    .. warning::
        This function is deprecated in favor of :func:`partialtorch.nn.functional.partial_interpolate`.
        This is equivalent with ``nn.functional.partial_interpolate(...)``.
    """
    warnings.warn("nn.functional.partial_upsample is deprecated. "
                  "Use nn.functional.partial_interpolate instead.")
    return partial_interpolate(input, size, scale_factor, mode, align_corners)


@_overload  # noqa: F811
def partial_interpolate(input: MaskedPair,
                        size: Optional[int] = None, scale_factor: Optional[List[float]] = None,
                        mode: str = 'nearest',
                        align_corners: Optional[bool] = None,
                        recompute_scale_factor: Optional[bool] = None,
                        antialias: bool = False) -> MaskedPair:  # noqa: F811,B950
    pass


@_overload  # noqa: F811
def partial_interpolate(input: MaskedPair,
                        size: Optional[List[int]] = None,
                        scale_factor: Optional[List[float]] = None,
                        mode: str = 'nearest',
                        align_corners: Optional[bool] = None,
                        recompute_scale_factor: Optional[bool] = None,
                        antialias: bool = False) -> MaskedPair:  # noqa: F811,B950
    pass


@_overload  # noqa: F811
def partial_interpolate(input: MaskedPair,
                        size: Optional[int] = None,
                        scale_factor: Optional[float] = None,
                        mode: str = 'nearest',
                        align_corners: Optional[bool] = None,
                        recompute_scale_factor: Optional[bool] = None,
                        antialias: bool = False) -> MaskedPair:  # noqa: F811,B950
    pass


@_overload  # noqa: F811
def partial_interpolate(input: MaskedPair,
                        size: Optional[List[int]] = None,
                        scale_factor: Optional[float] = None,
                        mode: str = "nearest",
                        align_corners: Optional[bool] = None,
                        recompute_scale_factor: Optional[bool] = None,
                        antialias: bool = False) -> MaskedPair:  # noqa: F811
    pass


def partial_interpolate(input: MaskedPair,
                        size: Optional[int] = None,
                        scale_factor: Optional[List[float]] = None,
                        mode: str = 'nearest',
                        align_corners: Optional[bool] = None,
                        recompute_scale_factor: Optional[bool] = None,
                        antialias: bool = False) -> MaskedPair:  # noqa: F811,B950
    r"""Partial variant of :func:`partialtorch.nn.functional.interpolate`.

    When mode is linear | bilinear | bicubic | trilinear, masked positions of the input
    will be filled with zeros before interpolation and any output with at least one valid
    interpolating operand will become valid.

    The behaviors for nearest | area | nearest-exact are identical to :func:`nn.functional.interpolate`.
    """
    if mode in ("nearest", "area", "nearest-exact"):
        if align_corners is not None:
            raise ValueError(
                "align_corners option can only be set with the "
                "interpolating modes: linear | bilinear | bicubic | trilinear"
            )
    else:
        if align_corners is None:
            align_corners = False

    dim = input.dim() - 2  # Number of spatial dimensions.

    # Process size and scale_factor.  Validate that exactly one is set.
    # Validate its length if it is a list, or expand it if it is a scalar.
    # After this block, exactly one of output_size and scale_factors will
    # be non-None, and it will be a list (or tuple).
    if size is not None and scale_factor is not None:
        raise ValueError("only one of size or scale_factor should be defined")
    elif size is not None:
        assert scale_factor is None
        scale_factors = None
        if isinstance(size, (list, tuple)):
            if len(size) != dim:
                raise ValueError(
                    "Input and output must have the same number of spatial dimensions, but got "
                    f"input with spatial dimensions of {list(input.shape[2:])} and output size of {size}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "output size in (o1, o2, ...,oK) format."
                )
            if not torch.jit.is_scripting():
                if not all(torch.nn.functional._is_integer(x) for x in size):
                    raise TypeError(
                        "expected size to be one of int or Tuple[int] or Tuple[int, int] or "
                        f"Tuple[int, int, int], but got size with types {[type(x) for x in size]}"
                    )
            output_size = size
        else:
            output_size = [size for _ in range(dim)]
    elif scale_factor is not None:
        assert size is None
        output_size = None
        if isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) != dim:
                raise ValueError(
                    "Input and scale_factor must have the same number of spatial dimensions, but "
                    f"got input with spatial dimensions of {list(input.shape[2:])} and "
                    f"scale_factor of shape {scale_factor}. "
                    "Please provide input tensor in (N, C, d1, d2, ...,dK) format and "
                    "scale_factor in (s1, s2, ...,sK) format."
                )
            scale_factors = scale_factor
        else:
            scale_factors = [scale_factor for _ in range(dim)]
    else:
        raise ValueError("either size or scale_factor should be defined")

    if recompute_scale_factor is not None and recompute_scale_factor and size is not None:
        raise ValueError("recompute_scale_factor is not meaningful with an explicit size.")

    # "area" mode always requires an explicit size rather than scale factor.
    # Re-use the recompute_scale_factor code path.
    if mode == "area" and output_size is None:
        recompute_scale_factor = True

    if recompute_scale_factor is not None and recompute_scale_factor:
        # We compute output_size here, then un-set scale_factors.
        # The C++ code will recompute it based on the (integer) output size.
        assert scale_factors is not None
        if not torch.jit.is_scripting() and torch._C._get_tracing_state():
            # make scale_factor a tensor in tracing so constant doesn't get baked in
            output_size = [
                (torch.floor((input.size(i + 2).float() * torch.tensor(scale_factors[i], dtype=torch.float32)).float()))
                for i in range(dim)
            ]
        elif torch.jit.is_scripting():
            output_size = [int(math.floor(float(input.size(i + 2)) * scale_factors[i]))
                           for i in range(dim)]
        else:
            output_size = [
                torch.sym_int(input.size(i + 2) * scale_factors[i])
                for i in range(dim)
            ]
        scale_factors = None

    if antialias and not (mode in ("bilinear", "bicubic") and input.ndim == 4):
        raise ValueError("Anti-alias option is only supported for bilinear and bicubic modes")

    if input.dim() == 3 and mode == "nearest":
        return partialtorch.ops.upsample_nearest1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest":
        return partialtorch.ops.upsample_nearest2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest":
        return partialtorch.ops.upsample_nearest3d(input, output_size, scale_factors)

    if input.dim() == 3 and mode == "nearest-exact":
        return partialtorch.ops._upsample_nearest_exact1d(input, output_size, scale_factors)
    if input.dim() == 4 and mode == "nearest-exact":
        return partialtorch.ops._upsample_nearest_exact2d(input, output_size, scale_factors)
    if input.dim() == 5 and mode == "nearest-exact":
        return partialtorch.ops._upsample_nearest_exact3d(input, output_size, scale_factors)

    if input.dim() == 3 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool1d(input, output_size)
    if input.dim() == 4 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool2d(input, output_size)
    if input.dim() == 5 and mode == "area":
        assert output_size is not None
        return adaptive_avg_pool3d(input, output_size)

    if input.dim() == 3 and mode == "linear":
        assert align_corners is not None
        return partialtorch.ops.partial_upsample_linear1d(input, output_size, align_corners, scale_factors)
    if input.dim() == 4 and mode == "bilinear":
        assert align_corners is not None
        if antialias:
            return partialtorch.ops._partial_upsample_bilinear2d_aa(input, output_size, align_corners, scale_factors)
        # No implementation for torch._decomp.decompositions.upsample_bilinear2d_vec
        return partialtorch.ops.partial_upsample_bilinear2d(input, output_size, align_corners, scale_factors)
    if input.dim() == 5 and mode == "trilinear":
        assert align_corners is not None
        return partialtorch.ops.partial_upsample_trilinear3d(input, output_size, align_corners, scale_factors)
    if input.dim() == 4 and mode == "bicubic":
        assert align_corners is not None
        if antialias:
            return partialtorch.ops._partial_upsample_bicubic2d_aa(input, output_size, align_corners, scale_factors)
        return partialtorch.ops.partial_upsample_bicubic2d(input, output_size, align_corners, scale_factors)

    if input.dim() == 3 and mode == "bilinear":
        raise NotImplementedError("Got 3D input, but bilinear mode needs 4D input")
    if input.dim() == 3 and mode == "trilinear":
        raise NotImplementedError("Got 3D input, but trilinear mode needs 5D input")
    if input.dim() == 4 and mode == "linear":
        raise NotImplementedError("Got 4D input, but linear mode needs 3D input")
    if input.dim() == 4 and mode == "trilinear":
        raise NotImplementedError("Got 4D input, but trilinear mode needs 5D input")
    if input.dim() == 5 and mode == "linear":
        raise NotImplementedError("Got 5D input, but linear mode needs 3D input")
    if input.dim() == 5 and mode == "bilinear":
        raise NotImplementedError("Got 5D input, but bilinear mode needs 4D input")

    raise NotImplementedError(
        "Input Error: Only 3D, 4D and 5D input Tensors supported"
        f" (got {input.dim()}D) for the modes: nearest | linear | bilinear | bicubic | trilinear | area | nearest-exact"
        f" (got {mode})"
    )


@_overload  # noqa: F811
def partial_upsample_nearest(input: MaskedPair,
                             size: Optional[int] = None,
                             scale_factor: Optional[float] = None) -> MaskedPair:  # noqa: F811
    pass


@_overload  # noqa: F811
def partial_upsample_nearest(input: MaskedPair,
                             size: Optional[List[int]] = None,
                             scale_factor: Optional[float] = None) -> MaskedPair:  # noqa: F811
    pass


def partial_upsample_nearest(input, size=None, scale_factor=None):  # noqa: F811
    r"""Partial variant of :func:`partialtorch.nn.functional.upsample_nearest`.

    .. warning::
        This function is deprecated in favor of :func:`partialtorch.nn.functional.partial_interpolate`.
        This is equivalent with ``nn.functional.partial_interpolate(..., mode='nearest')``.
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.partial_upsample_nearest is deprecated. "
                  "Use nn.functional.partial_interpolate instead.")
    return partial_interpolate(input, size, scale_factor, mode="nearest")


@_overload  # noqa: F811
def partial_upsample_bilinear(input: MaskedPair,
                              size: Optional[int] = None,
                              scale_factor: Optional[float] = None) -> MaskedPair:  # noqa: F811
    pass


@_overload  # noqa: F811
def partial_upsample_bilinear(input: MaskedPair,
                              size: Optional[List[int]] = None,
                              scale_factor: Optional[float] = None) -> MaskedPair:  # noqa: F811
    pass


@_overload  # noqa: F811
def partial_upsample_bilinear(input: MaskedPair,
                              size: Optional[int] = None,
                              scale_factor: Optional[List[float]] = None) -> MaskedPair:  # noqa: F811
    pass


@_overload  # noqa: F811
def partial_upsample_bilinear(input: MaskedPair,
                              size: Optional[List[int]] = None,
                              scale_factor: Optional[List[float]] = None) -> MaskedPair:  # noqa: F811
    pass


def partial_upsample_bilinear(input, size=None, scale_factor=None):  # noqa: F811
    r"""Partial variant of :func:`partialtorch.nn.functional.upsample_bilinear`.

    .. warning::
        This function is deprecated in favor of :func:`partialtorch.nn.functional.partial_interpolate`.
        This is equivalent with
        ``nn.functional.partial_interpolate(..., mode='bilinear', align_corners=True)``.
    """
    # DeprecationWarning is ignored by default
    warnings.warn("nn.functional.partial_upsample_bilinear is deprecated. "
                  "Use nn.functional.partial_interpolate instead.")
    return partial_interpolate(input, size, scale_factor, mode="bilinear", align_corners=True)


# TODO: grid_sample
pad = partialtorch.ops.pad
normalize = partialtorch.ops.normalize


def unfold(
        input: MaskedPair,
        kernel_size: BroadcastingList2[int],
        dilation: BroadcastingList2[int] = 1,
        padding: BroadcastingList2[int] = 0,
        stride: BroadcastingList2[int] = 1) -> MaskedPair:
    r"""See :func:`torch.nn.functional.unfold` for details.
    """
    return partialtorch.ops.im2col(input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))


def fold(
        input: MaskedPair,
        output_size: BroadcastingList2[int],
        kernel_size: BroadcastingList2[int],
        dilation: BroadcastingList2[int] = 1,
        padding: BroadcastingList2[int] = 0,
        stride: BroadcastingList2[int] = 1) -> MaskedPair:
    r"""See :func:`torch.nn.functional.fold` for details.
    """
    return partialtorch.ops.col2im(
        input, _pair(output_size), _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
