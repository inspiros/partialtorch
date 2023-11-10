import torch.nn.modules.fold

from partialtorch.types import MaskedPair
from .. import functional as partial_F

__all__ = ['Fold', 'Unfold']


class Fold(torch.nn.modules.fold.Fold):
    r"""See :class:`torch.nn.Fold` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.fold(input, self.output_size, self.kernel_size, self.dilation,
                              self.padding, self.stride)


class Unfold(torch.nn.modules.fold.Unfold):
    r"""See :class:`torch.nn.Unfold` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.unfold(input, self.kernel_size, self.dilation,
                                self.padding, self.stride)
