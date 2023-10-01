import torch

addcmul = torch.ops.partialtorch.addcmul
addcdiv = torch.ops.partialtorch.addcdiv

# -----------------------
# partial ternary
# -----------------------
partial_addmm = torch.ops.partialtorch.partial_addmm
partial_addbmm = torch.ops.partialtorch.partial_addbmm
partial_baddbmm = torch.ops.partialtorch.partial_baddbmm
partial_addmv = torch.ops.partialtorch.partial_addmv
partial_addr = torch.ops.partialtorch.partial_addr
partial_addcmul = torch.ops.partialtorch.partial_addcmul
partial_addcdiv = torch.ops.partialtorch.partial_addcdiv
