import torch.nn.modules.channelshuffle
from partialtorch.types import MaskedPair

from .. import functional as partial_F

__all__ = ['ChannelShuffle']


class ChannelShuffle(torch.nn.modules.channelshuffle.ChannelShuffle):
    r"""See :class:`torch.nn.ChannelShuffle` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.channel_shuffle(input, self.groups)
