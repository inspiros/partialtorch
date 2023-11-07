import torch.nn.modules.batchnorm

import partialtorch.nn.functional as partial_F
from partialtorch.types import MaskedPair

__all__ = [
    'BatchNorm1D',
    'BatchNorm2D',
    'BatchNorm3D',
]


class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def forward(self, input: MaskedPair) -> MaskedPair:
        self._check_input_dim(input.data)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        # Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return partial_F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )


class BatchNorm1D(_BatchNorm):
    r"""See :class:`torch.nn.BatchNorm1d` for details.
    """
    _check_input_dim = torch.nn.BatchNorm1d._check_input_dim


class BatchNorm2D(_BatchNorm):
    r"""See :class:`torch.nn.BatchNorm2d` for details.
    """
    _check_input_dim = torch.nn.BatchNorm2d._check_input_dim


class BatchNorm3D(_BatchNorm):
    r"""See :class:`torch.nn.BatchNorm3d` for details.
    """
    _check_input_dim = torch.nn.BatchNorm3d._check_input_dim