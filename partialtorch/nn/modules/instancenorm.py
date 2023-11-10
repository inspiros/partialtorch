import warnings

import partialtorch
import torch.nn.modules.instancenorm
from partialtorch.types import MaskedPair

from .. import functional as partial_F

__all__ = ['InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d']


class _InstanceNorm(torch.nn.modules.instancenorm._InstanceNorm):
    def _handle_no_batch_input(self, input: MaskedPair) -> MaskedPair:
        return partialtorch.squeeze(self._apply_instance_norm(partialtorch.unsqueeze(input, 0)), 0)

    def _apply_instance_norm(self, input: MaskedPair) -> MaskedPair:
        return partial_F.instance_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats, self.momentum, self.eps)

    def forward(self, input: MaskedPair) -> MaskedPair:
        self._check_input_dim(input)

        feature_dim = input.dim() - self._get_no_batch_dim()
        if input.size(feature_dim) != self.num_features:
            if self.affine:
                raise ValueError(
                    f"expected input's size at dim={feature_dim} to match num_features"
                    f" ({self.num_features}), but got: {input.size(feature_dim)}.")
            else:
                warnings.warn(f"input's size at dim={feature_dim} does not match num_features. "
                              "You can silence this warning by not passing in num_features, "
                              "which is not used because affine=False")

        if input.dim() == self._get_no_batch_dim():
            return self._handle_no_batch_input(input)

        return self._apply_instance_norm(input)


class InstanceNorm1d(_InstanceNorm):
    r"""See :class:`torch.nn.InstanceNorm1d` for details.
    """

    def _get_no_batch_dim(self):
        return 2

    def _check_input_dim(self, input):
        if input.dim() not in (2, 3):
            raise ValueError(f'expected 2D or 3D input (got {input.dim()}D input)')


class InstanceNorm2d(_InstanceNorm):
    r"""See :class:`torch.nn.InstanceNorm2d` for details.
    """

    def _get_no_batch_dim(self):
        return 3

    def _check_input_dim(self, input):
        if input.dim() not in (3, 4):
            raise ValueError(f'expected 3D or 4D input (got {input.dim()}D input)')


class InstanceNorm3d(_InstanceNorm):
    r"""See :class:`torch.nn.InstanceNorm3d` for details.
    """

    def _get_no_batch_dim(self):
        return 4

    def _check_input_dim(self, input):
        if input.dim() not in (4, 5):
            raise ValueError(f'expected 4D or 5D input (got {input.dim()}D input)')
