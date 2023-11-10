import torch.nn.modules.flatten

import partialtorch
from partialtorch import MaskedPair

__all__ = ['Flatten', 'Unflatten']


class Flatten(torch.nn.modules.flatten.Flatten):
    r"""See :class:`torch.nn.Flatten` for details.
    """

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partialtorch.flatten(input, self.start_dim, self.end_dim)


class Unflatten(torch.nn.modules.flatten.Unflatten):
    r"""See :class:`torch.nn.Unflatten` for details.
    """

    def _require_tuple_tuple(self, input):
        if (isinstance(input, tuple)):
            for idx, elem in enumerate(input):
                if not isinstance(elem, tuple):
                    raise TypeError("unflattened_size must be tuple of tuples, " +
                                    f"but found element of type {type(elem).__name__} at pos {idx}")
            return
        raise TypeError("unflattened_size must be a tuple of tuples, " +
                        f"but found type {type(input).__name__}")

    def _require_tuple_int(self, input):
        if (isinstance(input, (tuple, list))):
            for idx, elem in enumerate(input):
                if not isinstance(elem, int):
                    raise TypeError("unflattened_size must be tuple of ints, " +
                                    f"but found element of type {type(elem).__name__} at pos {idx}")
            return
        raise TypeError(f"unflattened_size must be a tuple of ints, but found type {type(input).__name__}")

    def forward(self, input: MaskedPair) -> MaskedPair:
        return partialtorch.unflatten(input, self.dim, self.unflattened_size)
