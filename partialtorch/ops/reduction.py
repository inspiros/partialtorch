import torch

# logical
all = torch.ops.partialtorch.all
any = torch.ops.partialtorch.any

# arithmetics
sum = torch.ops.partialtorch.sum
nansum = torch.ops.partialtorch.nansum
prod = torch.ops.partialtorch.prod
logsumexp = torch.ops.partialtorch.logsumexp
softmax = torch.ops.partialtorch.softmax
log_softmax = torch.ops.partialtorch.log_softmax
cumsum = torch.ops.partialtorch.cumsum
cumsum_ = torch.ops.partialtorch.cumsum_
cumprod = torch.ops.partialtorch.cumprod
cumprod_ = torch.ops.partialtorch.cumprod_
trace = torch.ops.partialtorch.trace

# statistics
mean = torch.ops.partialtorch.mean
nanmean = torch.ops.partialtorch.nanmean
median = torch.ops.partialtorch.median
nanmedian = torch.ops.partialtorch.nanmedian
var = torch.ops.partialtorch.var
std = torch.ops.partialtorch.std
norm = torch.ops.partialtorch.norm
linalg_norm = torch.ops.partialtorch.linalg_norm
linalg_vector_norm = torch.ops.partialtorch.linalg_vector_norm
linalg_matrix_norm = torch.ops.partialtorch.linalg_matrix_norm

# min/max
min = torch.ops.partialtorch.min
max = torch.ops.partialtorch.max
amin = torch.ops.partialtorch.amin
amax = torch.ops.partialtorch.amax
argmin = torch.ops.partialtorch.argmin
argmax = torch.ops.partialtorch.argmax
cummin = torch.ops.partialtorch.cummin
cummax = torch.ops.partialtorch.cummax

# torch.nn.functional
softmin = torch.ops.partialtorch.softmin
normalize = torch.ops.partialtorch.normalize
