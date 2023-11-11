import torch

# upsample_nearest
upsample_nearest1d = torch.ops.partialtorch.upsample_nearest1d
_upsample_nearest_exact1d = torch.ops.partialtorch._upsample_nearest_exact1d
upsample_nearest2d = torch.ops.partialtorch.upsample_nearest2d
_upsample_nearest_exact2d = torch.ops.partialtorch._upsample_nearest_exact2d
upsample_nearest3d = torch.ops.partialtorch.upsample_nearest3d
_upsample_nearest_exact3d = torch.ops.partialtorch._upsample_nearest_exact3d

# upsample_lerp
upsample_linear1d = torch.ops.partialtorch.upsample_linear1d
upsample_bilinear2d = torch.ops.partialtorch.upsample_bilinear2d
_upsample_bilinear2d_aa = torch.ops.partialtorch._upsample_bilinear2d_aa
upsample_trilinear3d = torch.ops.partialtorch.upsample_trilinear3d

# partial_upsample_lerp
partial_upsample_linear1d = torch.ops.partialtorch.partial_upsample_linear1d
partial_upsample_bilinear2d = torch.ops.partialtorch.partial_upsample_bilinear2d
_partial_upsample_bilinear2d_aa = torch.ops.partialtorch._partial_upsample_bilinear2d_aa
partial_upsample_trilinear3d = torch.ops.partialtorch.partial_upsample_trilinear3d

# upsample_bicubic
upsample_bicubic2d = torch.ops.partialtorch.upsample_bicubic2d
_upsample_bicubic2d_aa = torch.ops.partialtorch._upsample_bicubic2d_aa

# partial_upsample_bicubic
partial_upsample_bicubic2d = torch.ops.partialtorch.partial_upsample_bicubic2d
_partial_upsample_bicubic2d_aa = torch.ops.partialtorch._partial_upsample_bicubic2d_aa
