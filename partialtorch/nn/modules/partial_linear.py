import warnings

import torch.nn as nn

import partialtorch.nn.functional as partial_F
from partialtorch.types import MaskedPair

__all__ = [
    'PartialLinear'
]


# noinspection PyMethodOverriding
class PartialLinear(nn.Linear):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 scaled: bool = True,
                 device=None,
                 dtype=None):
        warnings.warn(f'{self.__class__.__name__} is an experimental module.', UserWarning)
        super(PartialLinear, self).__init__(in_features,
                                            out_features,
                                            bias,
                                            device,
                                            dtype)
        self.scaled = scaled

    def extra_repr(self) -> str:
        return super().extra_repr() + 'scaled={}'.format(self.scaled)

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.partial_linear(input, self.weight, self.bias, scaled=self.scaled)
