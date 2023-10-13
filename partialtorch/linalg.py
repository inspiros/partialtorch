r"""
Partial equivalence of :mod:`~torch.linalg`.
"""

import partialtorch

diagonal = partialtorch.ops.linalg_diagonal
partial_matmul = partialtorch.ops.linalg_partial_matmul
partial_multi_dot = partialtorch.ops.linalg_partial_multi_dot
partial_matrix_power = partialtorch.ops.linalg_partial_matrix_power
norm = partialtorch.ops.linalg_norm
vector_norm = partialtorch.ops.linalg_vector_norm
matrix_norm = partialtorch.ops.linalg_matrix_norm
