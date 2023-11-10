import torch

# max_pool
max_pool1d = torch.ops.partialtorch.max_pool1d
max_pool2d = torch.ops.partialtorch.max_pool2d
max_pool3d = torch.ops.partialtorch.max_pool3d
max_pool1d_with_indices = torch.ops.partialtorch.max_pool1d_with_indices
max_pool2d_with_indices = torch.ops.partialtorch.max_pool2d_with_indices
max_pool3d_with_indices = torch.ops.partialtorch.max_pool3d_with_indices
adaptive_max_pool1d = torch.ops.partialtorch.adaptive_max_pool1d
adaptive_max_pool2d = torch.ops.partialtorch.adaptive_max_pool2d
adaptive_max_pool3d = torch.ops.partialtorch.adaptive_max_pool3d

# fractional_max_pool
fractional_max_pool2d = torch.ops.partialtorch.fractional_max_pool2d
fractional_max_pool3d = torch.ops.partialtorch.fractional_max_pool3d
