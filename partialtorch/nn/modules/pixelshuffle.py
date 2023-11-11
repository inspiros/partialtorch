import torch.nn.modules.pixelshuffle
from partialtorch import MaskedPair

from .. import functional as partial_F

__all__ = ['PixelShuffle', 'PixelUnshuffle']


class PixelShuffle(torch.nn.modules.pixelshuffle.PixelShuffle):
    r"""See :class:`torch.nn.PixelShuffle` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.pixel_shuffle(input, self.upscale_factor)


class PixelUnshuffle(torch.nn.modules.pixelshuffle.PixelUnshuffle):
    r"""See :class:`torch.nn.PixelUnshuffle` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.pixel_unshuffle(input, self.downscale_factor)
