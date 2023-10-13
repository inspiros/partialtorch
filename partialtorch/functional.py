r"""
Partial equivalence of :mod:`~torch.functional`.

Note: All functions in this module are not :func:`~torch.jit.script`-able.
"""

from typing import Union, Optional

import torch

import partialtorch
from partialtorch.types import MaskedPair


def norm(input: MaskedPair,
         p: Optional[Union[float, str]] = "fro",
         dim=None,
         keepdim=False,
         out: MaskedPair = None,
         dtype=None) -> MaskedPair:  # noqa: F811
    r"""See :func:`~torch.functional.norm` for details.
    """
    # We don't do this for MPS or sparse tensors
    if input.layout != torch.strided and input.device.type not in \
            ("cpu", "cuda", "meta", torch.utils.backend_registration._privateuse1_backend_name):
        raise RuntimeError("partialtorch.functional.norm expects tensor with strided layout and "
                           "cpu, cuda, or meta device. Got "
                           f"input.layout={input.layout}, input.device={input.device}.")
    if out is not None:
        raise NotImplementedError("out ops are not yet supported")

    if dim is not None:
        if isinstance(dim, int):
            _dim = [dim]
        else:
            _dim = dim
    else:
        _dim = None  # type: ignore[assignment]

    if isinstance(p, str):
        if p == "fro" and (dim is None or isinstance(dim, int) or len(dim) <= 2):
            if out is None:
                return partialtorch.linalg.vector_norm(input, 2, _dim, keepdim, dtype=dtype)

        # Here we either call the nuclear norm, or we call matrix_norm with some arguments
        # that will throw an error
        if _dim is None:
            _dim = list(range(input.ndim))
        if out is None:
            return partialtorch.linalg.matrix_norm(input, p, _dim, keepdim, dtype=dtype)
    else:
        # NB. p should be Union[str, number], not Optional!
        _p = 2.0 if p is None else p
        if out is None:
            return partialtorch.linalg.vector_norm(input, _p, _dim, keepdim, dtype=dtype)
