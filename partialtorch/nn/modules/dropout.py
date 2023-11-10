import torch.nn.modules.dropout

from partialtorch.types import MaskedPair
from .. import functional as partial_F

__all__ = ['Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d', 'AlphaDropout', 'FeatureAlphaDropout']


class _DropoutNd(torch.nn.modules.dropout._DropoutNd):
    pass


class Dropout(_DropoutNd):
    r"""See :class:`torch.nn.Dropout` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.dropout(input, self.p, self.training, self.inplace)


class Dropout1d(_DropoutNd):
    r"""See :class:`torch.nn.Dropout1d` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.dropout1d(input, self.p, self.training, self.inplace)


class Dropout2d(_DropoutNd):
    r"""See :class:`torch.nn.Dropout2d` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.dropout2d(input, self.p, self.training, self.inplace)


class Dropout3d(_DropoutNd):
    r"""See :class:`torch.nn.Dropout3d` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.dropout3d(input, self.p, self.training, self.inplace)


class AlphaDropout(_DropoutNd):
    r"""See :class:`torch.nn.AlphaDropout` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.alpha_dropout(input, self.p, self.training)


class FeatureAlphaDropout(_DropoutNd):
    r"""See :class:`torch.nn.FeatureAlphaDropout` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.feature_alpha_dropout(input, self.p, self.training)
