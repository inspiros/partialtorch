import warnings

import torch.nn.modules.linear
from partialtorch.types import MaskedPair

from .. import functional as partial_F

__all__ = [
    'Identity',
    'PartialLinear',
    'PartialBilinear',
]


class Identity(torch.nn.modules.linear.Identity):
    r"""See :class:`torch.nn.Identity` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return input


# noinspection PyMethodOverriding
class PartialLinear(torch.nn.modules.linear.Linear):
    r"""See :class:`torch.nn.Linear` for details.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 scaled: bool = True,
                 device=None,
                 dtype=None):
        warnings.warn(f'{self.__class__.__name__} is an experimental module.', UserWarning)
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        self.scaled = scaled

    def extra_repr(self) -> str:
        return super().extra_repr() + 'scaled={}'.format(self.scaled)

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.partial_linear(input, self.weight, self.bias, scaled=self.scaled)


class PartialBilinear(torch.nn.modules.linear.Bilinear):
    r"""See :class:`torch.nn.Bilinear` for details.
    """

    def __init__(self,
                 in1_features: int,
                 in2_features: int,
                 out_features: int,
                 bias: bool = True,
                 scaled: bool = True,
                 device=None, dtype=None) -> None:
        warnings.warn(f'{self.__class__.__name__} is an experimental module.', UserWarning)
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(in1_features, in2_features, out_features, bias, **factory_kwargs)
        self.scaled = scaled

    def extra_repr(self) -> str:
        return super().extra_repr() + 'scaled={}'.format(self.scaled)

    def forward(self, input1: MaskedPair, input2: MaskedPair) -> MaskedPair:
        return partial_F.partial_bilinear(input1, input2, self.weight, self.bias, scaled=self.scaled)
