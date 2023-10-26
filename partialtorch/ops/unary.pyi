from typing import overload, Optional

from partialtorch.types import (
    _float, _int, _bool, Generator, Number,
    Tensor, MaskedPair, _MaskedPairOrTensor
)


# custom ops
def identity(self: _MaskedPairOrTensor) -> MaskedPair: ...


def fill_masked(self: _MaskedPairOrTensor,
                value: Number) -> MaskedPair: ...


def fill_masked_(self: _MaskedPairOrTensor,
                 value: Number) -> MaskedPair: ...


def to_tensor(self: _MaskedPairOrTensor,
              value: Number) -> Tensor: ...


def index_non_masked(self: _MaskedPairOrTensor) -> Tensor: ...


# torch ops
def abs(self: _MaskedPairOrTensor) -> MaskedPair: ...


def abs_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def absolute(self: _MaskedPairOrTensor) -> MaskedPair: ...


def absolute_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def acos(self: _MaskedPairOrTensor) -> MaskedPair: ...


def acos_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arccos(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arccos_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def acosh(self: _MaskedPairOrTensor) -> MaskedPair: ...


def acosh_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arccosh(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arccosh_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def alias(self: _MaskedPairOrTensor) -> MaskedPair: ...


def angle(self: _MaskedPairOrTensor) -> MaskedPair: ...


def asin(self: _MaskedPairOrTensor) -> MaskedPair: ...


def asin_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arcsin(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arcsin_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def asinh(self: _MaskedPairOrTensor) -> MaskedPair: ...


def asinh_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arcsinh(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arcsinh_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def atan(self: _MaskedPairOrTensor) -> MaskedPair: ...


def atan_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arctan(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arctan_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def atanh(self: _MaskedPairOrTensor) -> MaskedPair: ...


def atanh_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arctanh(self: _MaskedPairOrTensor) -> MaskedPair: ...


def arctanh_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_not(self: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_not_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def ceil(self: _MaskedPairOrTensor) -> MaskedPair: ...


def ceil_(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def clamp(self: _MaskedPairOrTensor,
          min: Optional[Number] = None,
          max: Optional[Number] = None) -> MaskedPair: ...


@overload
def clamp_(self: _MaskedPairOrTensor,
           min: Optional[Number] = None,
           max: Optional[Number] = None) -> MaskedPair: ...


@overload
def clamp(self: _MaskedPairOrTensor,
          min: Optional[Tensor] = None,
          max: Optional[Tensor] = None) -> MaskedPair: ...


@overload
def clamp_(self: _MaskedPairOrTensor,
           min: Optional[Tensor] = None,
           max: Optional[Tensor] = None) -> MaskedPair: ...


@overload
def clip(self: _MaskedPairOrTensor,
         min: Optional[Number] = None,
         max: Optional[Number] = None) -> MaskedPair: ...


@overload
def clip_(self: _MaskedPairOrTensor,
          min: Optional[Number] = None,
          max: Optional[Number] = None) -> MaskedPair: ...


@overload
def clip(self: _MaskedPairOrTensor,
         min: Optional[Tensor] = None,
         max: Optional[Tensor] = None) -> MaskedPair: ...


@overload
def clip_(self: _MaskedPairOrTensor,
          min: Optional[Tensor] = None,
          max: Optional[Tensor] = None) -> MaskedPair: ...


def conj_physical(self: _MaskedPairOrTensor) -> MaskedPair: ...


def conj_physical_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def cos(self: _MaskedPairOrTensor) -> MaskedPair: ...


def cos_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def cosh(self: _MaskedPairOrTensor) -> MaskedPair: ...


def cosh_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def deg2rad(self: _MaskedPairOrTensor) -> MaskedPair: ...


def deg2rad_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def rad2deg(self: _MaskedPairOrTensor) -> MaskedPair: ...


def rad2deg_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def dropout(self: _MaskedPairOrTensor,
            p: _float,
            train: _bool) -> MaskedPair: ...


def dropout_(self: _MaskedPairOrTensor,
             p: _float,
             train: _bool) -> MaskedPair: ...


def feature_dropout(self: _MaskedPairOrTensor,
                    p: _float,
                    train: _bool) -> MaskedPair: ...


def feature_dropout_(self: _MaskedPairOrTensor,
                     p: _float,
                     train: _bool) -> MaskedPair: ...


def alpha_dropout(self: _MaskedPairOrTensor,
                  p: _float,
                  train: _bool) -> MaskedPair: ...


def alpha_dropout_(self: _MaskedPairOrTensor,
                   p: _float,
                   train: _bool) -> MaskedPair: ...


def feature_alpha_dropout(self: _MaskedPairOrTensor,
                          p: _float,
                          train: _bool) -> MaskedPair: ...


def feature_alpha_dropout_(self: _MaskedPairOrTensor,
                           p: _float,
                           train: _bool) -> MaskedPair: ...


def erf(self: _MaskedPairOrTensor) -> MaskedPair: ...


def erf_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def erfc(self: _MaskedPairOrTensor) -> MaskedPair: ...


def erfc_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def erfinv(self: _MaskedPairOrTensor) -> MaskedPair: ...


def erfinv_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def exp(self: _MaskedPairOrTensor) -> MaskedPair: ...


def exp_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def exp2(self: _MaskedPairOrTensor) -> MaskedPair: ...


def exp2_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def expm1(self: _MaskedPairOrTensor) -> MaskedPair: ...


def expm1_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def fix(self: _MaskedPairOrTensor) -> MaskedPair: ...


def fix_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def floor(self: _MaskedPairOrTensor) -> MaskedPair: ...


def floor_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def frac(self: _MaskedPairOrTensor) -> MaskedPair: ...


def frac_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def lgamma(self: _MaskedPairOrTensor) -> MaskedPair: ...


def lgamma_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def digamma(self: _MaskedPairOrTensor) -> MaskedPair: ...


def digamma_(self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def polygamma(n: _int,
              self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def polygamma_(n: _int,
               self: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def polygamma(self: _MaskedPairOrTensor,
              n: _int) -> MaskedPair: ...


@overload
def polygamma_(self: _MaskedPairOrTensor,
               n: _int) -> MaskedPair: ...


def log(self: _MaskedPairOrTensor) -> MaskedPair: ...


def log_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def log10(self: _MaskedPairOrTensor) -> MaskedPair: ...


def log10_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def log1p(self: _MaskedPairOrTensor) -> MaskedPair: ...


def log1p_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def log2(self: _MaskedPairOrTensor) -> MaskedPair: ...


def log2_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def logit(self: _MaskedPairOrTensor,
          eps: Optional[_float] = None) -> MaskedPair: ...


def logit_(self: _MaskedPairOrTensor,
           eps: Optional[_float] = None) -> MaskedPair: ...


def i0(self: _MaskedPairOrTensor) -> MaskedPair: ...


def i0_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def isnan(self: _MaskedPairOrTensor) -> MaskedPair: ...


def isreal(self: _MaskedPairOrTensor) -> MaskedPair: ...


def isfinite(self: _MaskedPairOrTensor) -> MaskedPair: ...


def isinf(self: _MaskedPairOrTensor) -> MaskedPair: ...


def isposinf(self: _MaskedPairOrTensor) -> MaskedPair: ...


def isneginf(self: _MaskedPairOrTensor) -> MaskedPair: ...


def matrix_exp(self: _MaskedPairOrTensor) -> MaskedPair: ...


def matrix_power(self: _MaskedPairOrTensor,
                 n: _int) -> MaskedPair: ...


def nan_to_num(self: _MaskedPairOrTensor,
               nan: Optional[_float] = None,
               posinf: Optional[_float] = None,
               neginf: Optional[_float] = None) -> MaskedPair: ...


def nan_to_num_(self: _MaskedPairOrTensor,
                nan: Optional[_float] = None,
                posinf: Optional[_float] = None,
                neginf: Optional[_float] = None) -> MaskedPair: ...


def neg(self: _MaskedPairOrTensor) -> MaskedPair: ...


def neg_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def negative(self: _MaskedPairOrTensor) -> MaskedPair: ...


def negative_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def positive(self: _MaskedPairOrTensor) -> MaskedPair: ...


def reciprocal(self: _MaskedPairOrTensor) -> MaskedPair: ...


def reciprocal_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def round(self: _MaskedPairOrTensor) -> MaskedPair: ...


def round_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def rsqrt(self: _MaskedPairOrTensor) -> MaskedPair: ...


def rsqrt_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sign(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sign_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sgn(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sgn_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def signbit(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sin(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sin_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sinc(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sinc_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sinh(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sinh_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sqrt(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sqrt_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def square(self: _MaskedPairOrTensor) -> MaskedPair: ...


def square_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def tan(self: _MaskedPairOrTensor) -> MaskedPair: ...


def tan_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def tanh(self: _MaskedPairOrTensor) -> MaskedPair: ...


def tanh_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def hardtanh(self: _MaskedPairOrTensor,
             min_val: Number = -1,
             max_val: Number = 1) -> MaskedPair: ...


def hardtanh_(self: _MaskedPairOrTensor,
              min_val: Number = -1,
              max_val: Number = 1) -> MaskedPair: ...


def threshold(self: _MaskedPairOrTensor,
              threshold: Number,
              value: Number) -> MaskedPair: ...


def threshold_(self: _MaskedPairOrTensor,
               threshold: Number,
               value: Number) -> MaskedPair: ...


def trunc(self: _MaskedPairOrTensor) -> MaskedPair: ...


def trunc_(self: _MaskedPairOrTensor) -> MaskedPair: ...


# activations
def relu(self: _MaskedPairOrTensor) -> MaskedPair: ...


def relu_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def relu6(self: _MaskedPairOrTensor) -> MaskedPair: ...


def relu6_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def glu(self: _MaskedPairOrTensor,
        dim: _int = -1) -> MaskedPair: ...


def elu(self: _MaskedPairOrTensor,
        alpha: Number = 1,
        scale: Number = 1,
        input_scale: Number = 1) -> MaskedPair: ...


def elu_(self: _MaskedPairOrTensor,
         alpha: Number = 1,
         scale: Number = 1,
         input_scale: Number = 1) -> MaskedPair: ...


def selu(self: _MaskedPairOrTensor) -> MaskedPair: ...


def selu_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def celu(self: _MaskedPairOrTensor,
         alpha: Number = 1) -> MaskedPair: ...


def celu_(self: _MaskedPairOrTensor,
          alpha: Number = 1) -> MaskedPair: ...


def leaky_relu(self: _MaskedPairOrTensor,
               negative_slope: Number = 0.01) -> MaskedPair: ...


def leaky_relu_(self: _MaskedPairOrTensor,
                negative_slope: Number = 0.01) -> MaskedPair: ...


def prelu(self: _MaskedPairOrTensor,
          weight: Tensor) -> MaskedPair: ...


def rrelu(self: _MaskedPairOrTensor,
          lower: Number = 0.125,
          upper: Number = 0.3333333333333333,
          training: bool = False,
          generator: Optional[Generator] = None) -> MaskedPair: ...


def rrelu_(self: _MaskedPairOrTensor,
           lower: Number = 0.125,
           upper: Number = 0.3333333333333333,
           training: bool = False,
           generator: Optional[Generator] = None) -> MaskedPair: ...


def gelu(self: _MaskedPairOrTensor,
         approximate: str = "none") -> MaskedPair: ...


def gelu_(self: _MaskedPairOrTensor,
          approximate: str = "none") -> MaskedPair: ...


def hardshrink(self: _MaskedPairOrTensor,
               lambd: Number = 0.5) -> MaskedPair: ...


def softshrink(self: _MaskedPairOrTensor,
               lambd: Number = 0.5) -> MaskedPair: ...


def softplus(self: _MaskedPairOrTensor,
             beta: Number = 1,
             threshold: Number = 20) -> MaskedPair: ...


def sigmoid(self: _MaskedPairOrTensor) -> MaskedPair: ...


def sigmoid_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def hardsigmoid(self: _MaskedPairOrTensor) -> MaskedPair: ...


def hardsigmoid_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def log_sigmoid(self: _MaskedPairOrTensor) -> MaskedPair: ...


def silu(self: _MaskedPairOrTensor) -> MaskedPair: ...


def silu_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def mish(self: _MaskedPairOrTensor) -> MaskedPair: ...


def mish_(self: _MaskedPairOrTensor) -> MaskedPair: ...


def hardswish(self: _MaskedPairOrTensor) -> MaskedPair: ...


def hardswish_(self: _MaskedPairOrTensor) -> MaskedPair: ...


# torch.nn.functional
def dropout1d(self: _MaskedPairOrTensor,
              p: _float,
              train: _bool) -> MaskedPair: ...


def dropout1d_(self: _MaskedPairOrTensor,
               p: _float,
               train: _bool) -> MaskedPair: ...


def dropout2d(self: _MaskedPairOrTensor,
              p: _float,
              train: _bool) -> MaskedPair: ...


def dropout2d_(self: _MaskedPairOrTensor,
               p: _float,
               train: _bool) -> MaskedPair: ...


def dropout3d(self: _MaskedPairOrTensor,
              p: _float,
              train: _bool) -> MaskedPair: ...


def dropout3d_(self: _MaskedPairOrTensor,
               p: _float,
               train: _bool) -> MaskedPair: ...


def logsigmoid(self: _MaskedPairOrTensor) -> MaskedPair: ...


def softsign(self: _MaskedPairOrTensor) -> MaskedPair: ...


def tanhshrink(self: _MaskedPairOrTensor) -> MaskedPair: ...


# properties
def dim(self: MaskedPair) -> _int: ...


def dense_dim(self: MaskedPair) -> _int: ...


def size(self: MaskedPair,
         dim: _int) -> _int: ...


def requires_grad_(self: _MaskedPairOrTensor,
                   requires_grad: _bool = True) -> MaskedPair: ...


# ----------------------
# partial unary
# ----------------------
def linalg_partial_matrix_power(self: _MaskedPairOrTensor,
                                n: _int) -> MaskedPair: ...
