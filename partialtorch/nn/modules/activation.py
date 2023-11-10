import partialtorch
import torch.nn.modules.activation
from partialtorch.types import MaskedPair

from .. import functional as partial_F

__all__ = ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid', 'Tanh',
           'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU', 'Hardshrink', 'LeakyReLU',
           'LogSigmoid', 'Softplus', 'Softshrink', 'PReLU', 'Softsign', 'Tanhshrink',
           'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']


class Threshold(torch.nn.modules.activation.Threshold):
    r"""See :class:`torch.nn.Threshold` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.threshold(input, self.threshold, self.value, self.inplace)


class ReLU(torch.nn.modules.activation.ReLU):
    r"""See :class:`torch.nn.ReLU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.relu(input, inplace=self.inplace)


class RReLU(torch.nn.modules.activation.RReLU):
    r"""See :class:`torch.nn.RReLU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.rrelu(input, self.lower, self.upper, self.training, self.inplace)


class Hardtanh(torch.nn.modules.activation.Hardtanh):
    r"""See :class:`torch.nn.Hardtanh` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.hardtanh(input, self.min_val, self.max_val, self.inplace)


class ReLU6(torch.nn.modules.activation.ReLU6):
    r"""See :class:`torch.nn.ReLU6` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.relu6(input, self.inplace)


class Sigmoid(torch.nn.modules.activation.Sigmoid):
    r"""See :class:`torch.nn.Sigmoid` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partialtorch.sigmoid(input)


class Hardsigmoid(torch.nn.modules.activation.Hardsigmoid):
    r"""See :class:`torch.nn.Hardsigmoid` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.hardsigmoid(input, self.inplace)


class Tanh(torch.nn.modules.activation.Tanh):
    r"""See :class:`torch.nn.Tanh` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partialtorch.tanh(input)


class SiLU(torch.nn.modules.activation.SiLU):
    r"""See :class:`torch.nn.SiLU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.silu(input, inplace=self.inplace)


class Mish(torch.nn.modules.activation.Mish):
    r"""See :class:`torch.nn.Mish` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.mish(input, inplace=self.inplace)


class Hardswish(torch.nn.modules.activation.Hardswish):
    r"""See :class:`torch.nn.Hardswish` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.hardswish(input, self.inplace)


class ELU(torch.nn.modules.activation.ELU):
    r"""See :class:`torch.nn.ELU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.elu(input, self.alpha, self.inplace)


class CELU(torch.nn.modules.activation.CELU):
    r"""See :class:`torch.nn.CELU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.celu(input, self.alpha, self.inplace)


class SELU(torch.nn.modules.activation.SELU):
    r"""See :class:`torch.nn.SELU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.selu(input, self.inplace)


class GLU(torch.nn.modules.activation.GLU):
    r"""See :class:`torch.nn.GLU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.glu(input, self.dim)


class GELU(torch.nn.modules.activation.GELU):
    r"""See :class:`torch.nn.GELU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.gelu(input, approximate=self.approximate)


class Hardshrink(torch.nn.modules.activation.Hardshrink):
    r"""See :class:`torch.nn.Hardshrink` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.hardshrink(input, self.lambd)


class LeakyReLU(torch.nn.modules.activation.LeakyReLU):
    r"""See :class:`torch.nn.LeakyReLU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.leaky_relu(input, self.negative_slope, self.inplace)


class LogSigmoid(torch.nn.modules.activation.LogSigmoid):
    r"""See :class:`torch.nn.LogSigmoid` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.logsigmoid(input)


class Softplus(torch.nn.modules.activation.Softplus):
    r"""See :class:`torch.nn.Softplus` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.softplus(input, self.beta, self.threshold)


class Softshrink(torch.nn.modules.activation.Softshrink):
    r"""See :class:`torch.nn.Softshrink` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.softshrink(input, self.lambd)


# TODO: MultiheadAttention

class PReLU(torch.nn.modules.activation.PReLU):
    r"""See :class:`torch.nn.PReLU` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.prelu(input, self.weight)


class Softsign(torch.nn.modules.activation.Softsign):
    r"""See :class:`torch.nn.Softsign` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.softsign(input)


class Tanhshrink(torch.nn.modules.activation.Tanhshrink):
    r"""See :class:`torch.nn.Tanhshrink` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.tanhshrink(input)


class Softmin(torch.nn.modules.activation.Softmin):
    r"""See :class:`torch.nn.Softmin` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.softmin(input, self.dim)


class Softmax(torch.nn.modules.activation.Softmax):
    r"""See :class:`torch.nn.Softmax` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.softmax(input, self.dim)


class Softmax2d(torch.nn.modules.activation.Softmax2d):
    r"""See :class:`torch.nn.Softmax2d` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        if input.dim() not in (3, 4):
            raise ValueError(
                f"Softmax2d: expected input to be 3D or 4D, got {input.dim()}D instead"
            )
        return partial_F.softmax(input, -3)


class LogSoftmax(torch.nn.modules.activation.LogSoftmax):
    r"""See :class:`torch.nn.LogSoftmax` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partial_F.log_softmax(input, self.dim)
