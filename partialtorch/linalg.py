r"""
Partial equivalence of :mod:`~torch.linalg`.
"""

import partialtorch

norm = partialtorch.ops.linalg_norm
vector_norm = partialtorch.ops.linalg_vector_norm
matrix_norm = partialtorch.ops.linalg_matrix_norm
