import torch

# logical
bitwise_and = torch.ops.partialtorch.bitwise_and
bitwise_and_ = torch.ops.partialtorch.bitwise_and_
bitwise_or = torch.ops.partialtorch.bitwise_or
bitwise_or_ = torch.ops.partialtorch.bitwise_or_
bitwise_xor = torch.ops.partialtorch.bitwise_xor
bitwise_xor_ = torch.ops.partialtorch.bitwise_xor_
bitwise_left_shift = torch.ops.partialtorch.bitwise_left_shift
bitwise_left_shift_ = torch.ops.partialtorch.bitwise_left_shift_
bitwise_right_shift = torch.ops.partialtorch.bitwise_right_shift
bitwise_right_shift_ = torch.ops.partialtorch.bitwise_right_shift_

# python logical magic methods
__and__ = torch.ops.partialtorch.__and__
__iand__ = torch.ops.partialtorch.__iand__
__or__ = torch.ops.partialtorch.__or__
__ior__ = torch.ops.partialtorch.__ior__
__xor__ = torch.ops.partialtorch.__xor__
__ixor__ = torch.ops.partialtorch.__ixor__
__lshift__ = torch.ops.partialtorch.__lshift__
__ilshift__ = torch.ops.partialtorch.__ilshift__
__rshift__ = torch.ops.partialtorch.__rshift__
__irshift__ = torch.ops.partialtorch.__irshift__

# arithmetics
add = torch.ops.partialtorch.add
add_ = torch.ops.partialtorch.add_
sub = torch.ops.partialtorch.sub
sub_ = torch.ops.partialtorch.sub_
subtract = torch.ops.partialtorch.subtract
subtract_ = torch.ops.partialtorch.subtract_
mul = torch.ops.partialtorch.mul
mul_ = torch.ops.partialtorch.mul_
multiply = torch.ops.partialtorch.multiply
multiply_ = torch.ops.partialtorch.multiply_
div = torch.ops.partialtorch.div
div_ = torch.ops.partialtorch.div_
divide = torch.ops.partialtorch.divide
divide_ = torch.ops.partialtorch.divide_
floor_divide = torch.ops.partialtorch.floor_divide
floor_divide_ = torch.ops.partialtorch.floor_divide_
true_divide = torch.ops.partialtorch.true_divide
true_divide_ = torch.ops.partialtorch.true_divide_
fmod = torch.ops.partialtorch.fmod
fmod_ = torch.ops.partialtorch.fmod_
remainder = torch.ops.partialtorch.remainder
remainder_ = torch.ops.partialtorch.remainder_
pow = torch.ops.partialtorch.pow
pow_ = torch.ops.partialtorch.pow_
float_power = torch.ops.partialtorch.float_power
float_power_ = torch.ops.partialtorch.float_power_
atan2 = torch.ops.partialtorch.atan2
atan2_ = torch.ops.partialtorch.atan2_
arctan2 = torch.ops.partialtorch.arctan2
arctan2_ = torch.ops.partialtorch.arctan2_
logaddexp = torch.ops.partialtorch.logaddexp
logaddexp2 = torch.ops.partialtorch.logaddexp2
nextafter = torch.ops.partialtorch.nextafter
nextafter_ = torch.ops.partialtorch.nextafter_
ldexp = torch.ops.partialtorch.ldexp
ldexp_ = torch.ops.partialtorch.ldexp_
lerp = torch.ops.partialtorch.lerp
lerp_ = torch.ops.partialtorch.lerp_
dist = torch.ops.partialtorch.dist

# comparison
isclose = torch.ops.partialtorch.isclose
equal = torch.ops.partialtorch.equal
eq = torch.ops.partialtorch.eq
eq_ = torch.ops.partialtorch.eq_
ne = torch.ops.partialtorch.ne
ne_ = torch.ops.partialtorch.ne_
not_equal = torch.ops.partialtorch.not_equal
not_equal_ = torch.ops.partialtorch.not_equal_
lt = torch.ops.partialtorch.lt
lt_ = torch.ops.partialtorch.lt_
less = torch.ops.partialtorch.less
less_ = torch.ops.partialtorch.less_
gt = torch.ops.partialtorch.gt
gt_ = torch.ops.partialtorch.gt_
greater = torch.ops.partialtorch.greater
greater_ = torch.ops.partialtorch.greater_
le = torch.ops.partialtorch.le
le_ = torch.ops.partialtorch.le_
less_equal = torch.ops.partialtorch.less_equal
less_equal_ = torch.ops.partialtorch.less_equal_
ge = torch.ops.partialtorch.ge
ge_ = torch.ops.partialtorch.ge_
greater_equal = torch.ops.partialtorch.greater_equal
greater_equal_ = torch.ops.partialtorch.greater_equal_

# min max
min = torch.ops.partialtorch.min
max = torch.ops.partialtorch.max
minimum = torch.ops.partialtorch.minimum
maxium = torch.ops.partialtorch.maxium
fmin = torch.ops.partialtorch.fmin
fmax = torch.ops.partialtorch.fmax

# ----------------------
# partial bitwise binary
# ----------------------
# logical
partial_bitwise_and = torch.ops.partialtorch.partial_bitwise_and
partial_bitwise_and_ = torch.ops.partialtorch.partial_bitwise_and_
partial_bitwise_or = torch.ops.partialtorch.partial_bitwise_or
partial_bitwise_or_ = torch.ops.partialtorch.partial_bitwise_or_
partial_bitwise_xor = torch.ops.partialtorch.partial_bitwise_xor
partial_bitwise_xor_ = torch.ops.partialtorch.partial_bitwise_xor_

# arithmetics
partial_add = torch.ops.partialtorch.partial_add
partial_add_ = torch.ops.partialtorch.partial_add_
partial_sub = torch.ops.partialtorch.partial_sub
partial_sub_ = torch.ops.partialtorch.partial_sub_
partial_subtract = torch.ops.partialtorch.partial_subtract
partial_subtract_ = torch.ops.partialtorch.partial_subtract_
partial_mul = torch.ops.partialtorch.partial_mul
partial_mul_ = torch.ops.partialtorch.partial_mul_
partial_multiply = torch.ops.partialtorch.partial_multiply
partial_multiply_ = torch.ops.partialtorch.partial_multiply_
partial_div = torch.ops.partialtorch.partial_div
partial_div_ = torch.ops.partialtorch.partial_div_
partial_divide = torch.ops.partialtorch.partial_divide
partial_divide_ = torch.ops.partialtorch.partial_divide_
partial_floor_divide = torch.ops.partialtorch.partial_floor_divide
partial_floor_divide_ = torch.ops.partialtorch.partial_floor_divide_
partial_true_divide = torch.ops.partialtorch.partial_true_divide
partial_true_divide_ = torch.ops.partialtorch.partial_true_divide_
partial_logaddexp = torch.ops.partialtorch.partial_logaddexp
partial_logaddexp2 = torch.ops.partialtorch.partial_logaddexp2

# comparison
partial_isclose = torch.ops.partialtorch.partial_isclose
partial_equal = torch.ops.partialtorch.partial_equal
partial_eq = torch.ops.partialtorch.partial_eq
partial_eq_ = torch.ops.partialtorch.partial_eq_
partial_ne = torch.ops.partialtorch.partial_ne
partial_ne_ = torch.ops.partialtorch.partial_ne_
partial_not_equal = torch.ops.partialtorch.partial_not_equal
partial_not_equal_ = torch.ops.partialtorch.partial_not_equal_

# ----------------------
# partial binary
# ----------------------
partial_ger = torch.ops.partialtorch.partial_ger
partial_mm = torch.ops.partialtorch.partial_mm
partial_bmm = torch.ops.partialtorch.partial_bmm
partial_matmul = torch.ops.partialtorch.partial_matmul
linalg_partial_matmul = torch.ops.partialtorch.linalg_partial_matmul
partial_mv = torch.ops.partialtorch.partial_mv
partial_inner = torch.ops.partialtorch.partial_inner
partial_outer = torch.ops.partialtorch.partial_outer
