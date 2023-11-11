from typing import Optional

import torch.nn.modules.upsampling
from torch.nn.common_types import _size_2_t, _ratio_2_t

from partialtorch.types import MaskedPair
from .. import functional as partial_F

__all__ = [
    'Upsample', 'UpsamplingNearest2d', 'UpsamplingBilinear2d',
    'PartialUpsample', 'PartialUpsamplingBilinear2d']


class Upsample(torch.nn.modules.upsampling.Upsample):
    r"""See :class:`torch.nn.Upsample` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
                                     recompute_scale_factor=self.recompute_scale_factor)


class UpsamplingNearest2d(Upsample):
    r"""See :class:`torch.nn.UpsamplingNearest2d` for details.
    """

    def __init__(self, size: Optional[_size_2_t] = None, scale_factor: Optional[_ratio_2_t] = None) -> None:
        super().__init__(size, scale_factor, mode='nearest')


class UpsamplingBilinear2d(Upsample):
    r"""See :class:`torch.nn.UpsamplingBilinear2d` for details.
    """

    def __init__(self, size: Optional[_size_2_t] = None, scale_factor: Optional[_ratio_2_t] = None) -> None:
        super().__init__(size, scale_factor, mode='bilinear', align_corners=True)


class PartialUpsample(Upsample):
    r"""Partial variant of :class:`partialtorch.nn.Upsample`.

    When mode is linear | bilinear | bicubic | trilinear, masked positions of the input
    will be filled with zeros before interpolation and any output with at least one valid
    interpolating operand will become valid.

    The behaviors for nearest | area | nearest-exact are identical to :class:`partialtorch.nn.Upsample`.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.partial_interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners,
                                             recompute_scale_factor=self.recompute_scale_factor)


class PartialUpsamplingBilinear2d(PartialUpsample):
    r"""Partial variant of :class:`partialtorch.nn.UpsamplingBilinear2d`.
    See :class:`partialtorch.nn.PartialUpsample` for details.
    """

    def __init__(self, size: Optional[_size_2_t] = None, scale_factor: Optional[_ratio_2_t] = None) -> None:
        super().__init__(size, scale_factor, mode='bilinear', align_corners=True)
