import torch

# to
to = torch.ops.partialtorch.to
cpu = torch.ops.partialtorch.cpu
cuda = torch.ops.partialtorch.cuda

# one to one
clone = torch.ops.partialtorch.clone
contiguous = torch.ops.partialtorch.contiguous
detach = torch.ops.partialtorch.detach
detach_copy = torch.ops.partialtorch.detach_copy
diag = torch.ops.partialtorch.diag
diag_embed = torch.ops.partialtorch.diag_embed
diagflat = torch.ops.partialtorch.diagflat
diagonal = torch.ops.partialtorch.diagonal
linalg_diagonal = torch.ops.partialtorch.linalg_diagonal
narrow = torch.ops.partialtorch.narrow
narrow_copy = torch.ops.partialtorch.narrow_copy
select = torch.ops.partialtorch.select
repeat = torch.ops.partialtorch.repeat
repeat_interleave = torch.ops.partialtorch.repeat_interleave
tile = torch.ops.partialtorch.tile
ravel = torch.ops.partialtorch.ravel
flatten = torch.ops.partialtorch.flatten
unflatten = torch.ops.partialtorch.unflatten
broadcast_to = torch.ops.partialtorch.broadcast_to
expand = torch.ops.partialtorch.expand
expand_as = torch.ops.partialtorch.expand_as
reshape = torch.ops.partialtorch.reshape
reshape_as = torch.ops.partialtorch.reshape_as
view = torch.ops.partialtorch.view
view_as = torch.ops.partialtorch.view_as
squeeze = torch.ops.partialtorch.squeeze
squeeze_ = torch.ops.partialtorch.squeeze_
unsqueeze = torch.ops.partialtorch.unsqueeze
unsqueeze_ = torch.ops.partialtorch.unsqueeze_
matrix_H = torch.ops.partialtorch.matrix_H
moveaxis = torch.ops.partialtorch.moveaxis
moveaxes = torch.ops.partialtorch.moveaxes
movedim = torch.ops.partialtorch.movedim
movedims = torch.ops.partialtorch.movedims
swapaxis = torch.ops.partialtorch.swapaxis
swapaxis_ = torch.ops.partialtorch.swapaxis_
swapaxes = torch.ops.partialtorch.swapaxes
swapaxes_ = torch.ops.partialtorch.swapaxes_
swapdim = torch.ops.partialtorch.swapdim
swapdim_ = torch.ops.partialtorch.swapdim_
swapdims = torch.ops.partialtorch.swapdims
swapdims_ = torch.ops.partialtorch.swapdims_
t = torch.ops.partialtorch.t
t_ = torch.ops.partialtorch.t_
transpose = torch.ops.partialtorch.transpose
transpose_ = torch.ops.partialtorch.transpose_
permute = torch.ops.partialtorch.permute
permute_copy = torch.ops.partialtorch.permute_copy
take = torch.ops.partialtorch.take
take_along_dim = torch.ops.partialtorch.take_along_dim
gather = torch.ops.partialtorch.gather
unfold = torch.ops.partialtorch.unfold

# one to many
chunk = torch.ops.partialtorch.chunk
split = torch.ops.partialtorch.split
split_copy = torch.ops.partialtorch.split_copy
split_with_sizes = torch.ops.partialtorch.split_with_sizes
split_with_sizes_copy = torch.ops.partialtorch.split_with_sizes_copy
dsplit = torch.ops.partialtorch.dsplit
hsplit = torch.ops.partialtorch.hsplit
vsplit = torch.ops.partialtorch.vsplit

# many to one
cat = torch.ops.partialtorch.cat
row_stack = torch.ops.partialtorch.row_stack
column_stack = torch.ops.partialtorch.column_stack
hstack = torch.ops.partialtorch.hstack
vstack = torch.ops.partialtorch.vstack

# many to many
broadcast_tensors = torch.ops.partialtorch.broadcast_tensors
meshgrid = torch.ops.partialtorch.meshgrid

# one to one/many to many
atleast_1d = torch.ops.partialtorch.atleast_1d
atleast_2d = torch.ops.partialtorch.atleast_2d
atleast_3d = torch.ops.partialtorch.atleast_3d
