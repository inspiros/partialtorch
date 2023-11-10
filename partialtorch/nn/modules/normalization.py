import torch.nn.modules.normalization
from partialtorch.types import MaskedPair

from .. import functional as partial_F

__all__ = ['LayerNorm']


# TODO: LocalResponseNorm
# TODO: CrossMapLRN2d


class LayerNorm(torch.nn.modules.normalization.LayerNorm):
    r"""See :class:`torch.nn.LayerNorm` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

# TODO: GroupNorm
# TODO: ContrastiveNorm2d
# TODO: DivisiveNorm2d
# TODO: SubtractiveNorm2d
