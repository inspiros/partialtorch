from typing import overload, Optional

from partialtorch.types import (
    _float, _bool, Number,
    Tensor, MaskedPair, _MaskedPairOrTensor
)


# ----------------------
# bitwise binary
# ----------------------
# logical
def bitwise_and(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_and_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_or(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_or_(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_xor(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_xor_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_left_shift(self: _MaskedPairOrTensor,
                       other: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_left_shift_(self: _MaskedPairOrTensor,
                        other: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_right_shift(self: _MaskedPairOrTensor,
                        other: _MaskedPairOrTensor) -> MaskedPair: ...


def bitwise_right_shift_(self: _MaskedPairOrTensor,
                         other: _MaskedPairOrTensor) -> MaskedPair: ...


# python logical magic methods
def __and__(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor) -> MaskedPair: ...


def __iand__(self: _MaskedPairOrTensor,
             other: _MaskedPairOrTensor) -> MaskedPair: ...


def __or__(self: _MaskedPairOrTensor,
           other: _MaskedPairOrTensor) -> MaskedPair: ...


def __ior__(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor) -> MaskedPair: ...


def __xor__(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor) -> MaskedPair: ...


def __ixor__(self: _MaskedPairOrTensor,
             other: _MaskedPairOrTensor) -> MaskedPair: ...


def __lshift__(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def __ilshift__(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def __rshift__(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def __irshift__(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


# arithmetics
def add(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor,
        *,
        alpha: Number = 1) -> MaskedPair: ...


def add_(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor,
         *,
         alpha: Number = 1) -> MaskedPair: ...


def sub(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor,
        *,
        alpha: Number = 1) -> MaskedPair: ...


def sub_(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor,
         *,
         alpha: Number = 1) -> MaskedPair: ...


def subtract(self: _MaskedPairOrTensor,
             other: _MaskedPairOrTensor,
             *,
             alpha: Number = 1) -> MaskedPair: ...


def subtract_(self: _MaskedPairOrTensor,
              other: _MaskedPairOrTensor,
              *,
              alpha: Number = 1) -> MaskedPair: ...


def mul(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def mul_(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor) -> MaskedPair: ...


def multiply(self: _MaskedPairOrTensor,
             other: _MaskedPairOrTensor) -> MaskedPair: ...


def multiply_(self: _MaskedPairOrTensor,
              other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def div(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def div_(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def div(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor,
        *,
        rounding_mode: Optional[str] = None) -> MaskedPair: ...


@overload
def div_(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor,
         *,
         rounding_mode: Optional[str] = None) -> MaskedPair: ...


@overload
def divide(self: _MaskedPairOrTensor,
           other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def divide_(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def divide(self: _MaskedPairOrTensor,
           other: _MaskedPairOrTensor,
           *,
           rounding_mode: Optional[str] = None) -> MaskedPair: ...


@overload
def divide_(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor,
            *,
            rounding_mode: Optional[str] = None) -> MaskedPair: ...


def floor_divide(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor) -> MaskedPair: ...


def floor_divide_(self: _MaskedPairOrTensor,
                  other: _MaskedPairOrTensor) -> MaskedPair: ...


def true_divide(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def true_divide_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor) -> MaskedPair: ...


def fmod(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor) -> MaskedPair: ...


def fmod_(self: _MaskedPairOrTensor,
          other: _MaskedPairOrTensor) -> MaskedPair: ...


def remainder(self: _MaskedPairOrTensor,
              other: _MaskedPairOrTensor) -> MaskedPair: ...


def remainder_(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def pow(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def pow_(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor) -> MaskedPair: ...


def float_power(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def float_power_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor) -> MaskedPair: ...


def atan2(self: _MaskedPairOrTensor,
          other: _MaskedPairOrTensor) -> MaskedPair: ...


def atan2_(self: _MaskedPairOrTensor,
           other: _MaskedPairOrTensor) -> MaskedPair: ...


def arctan2(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor) -> MaskedPair: ...


def arctan2_(self: _MaskedPairOrTensor,
             other: _MaskedPairOrTensor) -> MaskedPair: ...


def logaddexp(self: _MaskedPairOrTensor,
              other: _MaskedPairOrTensor) -> MaskedPair: ...


def logaddexp2(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def nextafter(self: _MaskedPairOrTensor,
              other: _MaskedPairOrTensor) -> MaskedPair: ...


def nextafter_(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def ldexp(self: _MaskedPairOrTensor,
          other: _MaskedPairOrTensor) -> MaskedPair: ...


def ldexp_(self: _MaskedPairOrTensor,
           other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def lerp(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor,
         weight: Number) -> MaskedPair: ...


@overload
def lerp_(self: _MaskedPairOrTensor,
          other: _MaskedPairOrTensor,
          weight: Number) -> MaskedPair: ...


@overload
def lerp(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor,
         weight: Tensor) -> MaskedPair: ...


@overload
def lerp_(self: _MaskedPairOrTensor,
          other: _MaskedPairOrTensor,
          weight: Tensor) -> MaskedPair: ...


# comparison
def allclose(self: _MaskedPairOrTensor,
             other: _MaskedPairOrTensor,
             rtol: _float = 1e-05,
             atol: _float = 1e-08,
             equal_nan: _bool = False) -> _bool: ...


def isclose(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor,
            rtol: _float = 1e-05,
            atol: _float = 1e-08,
            equal_nan: _bool = False) -> MaskedPair: ...


def equal(self: _MaskedPairOrTensor,
          other: _MaskedPairOrTensor) -> _bool: ...


def eq(self: _MaskedPairOrTensor,
       other: _MaskedPairOrTensor) -> MaskedPair: ...


def eq_(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def ne(self: _MaskedPairOrTensor,
       other: _MaskedPairOrTensor) -> MaskedPair: ...


def ne_(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def not_equal(self: _MaskedPairOrTensor,
              other: _MaskedPairOrTensor) -> MaskedPair: ...


def not_equal_(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def lt(self: _MaskedPairOrTensor,
       other: _MaskedPairOrTensor) -> MaskedPair: ...


def lt_(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def less(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor) -> MaskedPair: ...


def less_(self: _MaskedPairOrTensor,
          other: _MaskedPairOrTensor) -> MaskedPair: ...


def gt(self: _MaskedPairOrTensor,
       other: _MaskedPairOrTensor) -> MaskedPair: ...


def gt_(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def greater(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor) -> MaskedPair: ...


def greater_(self: _MaskedPairOrTensor,
             other: _MaskedPairOrTensor) -> MaskedPair: ...


def le(self: _MaskedPairOrTensor,
       other: _MaskedPairOrTensor) -> MaskedPair: ...


def le_(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def less_equal(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def less_equal_(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def ge(self: _MaskedPairOrTensor,
       other: _MaskedPairOrTensor) -> MaskedPair: ...


def ge_(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def greater_equal(self: _MaskedPairOrTensor,
                  other: _MaskedPairOrTensor) -> MaskedPair: ...


def greater_equal_(self: _MaskedPairOrTensor,
                   other: _MaskedPairOrTensor) -> MaskedPair: ...


# min max
def min(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def max(self: _MaskedPairOrTensor,
        other: _MaskedPairOrTensor) -> MaskedPair: ...


def minimum(self: _MaskedPairOrTensor,
            other: _MaskedPairOrTensor) -> MaskedPair: ...


def maxium(self: _MaskedPairOrTensor,
           other: _MaskedPairOrTensor) -> MaskedPair: ...


def fmin(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor) -> MaskedPair: ...


def fmax(self: _MaskedPairOrTensor,
         other: _MaskedPairOrTensor) -> MaskedPair: ...


# ----------------------
# partial bitwise binary
# ----------------------
# logical
def partial_bitwise_and(self: _MaskedPairOrTensor,
                        other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_bitwise_and_(self: _MaskedPairOrTensor,
                         other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_bitwise_or(self: _MaskedPairOrTensor,
                       other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_bitwise_or_(self: _MaskedPairOrTensor,
                        other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_bitwise_xor(self: _MaskedPairOrTensor,
                        other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_bitwise_xor_(self: _MaskedPairOrTensor,
                         other: _MaskedPairOrTensor) -> MaskedPair: ...


# arithmetics
@overload
def partial_add(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor,
                *,
                alpha: Number = 1) -> MaskedPair: ...


@overload
def partial_add_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor,
                 *,
                 alpha: Number = 1) -> MaskedPair: ...


@overload
def partial_sub(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor,
                *,
                alpha: Number = 1) -> MaskedPair: ...


@overload
def partial_sub_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor,
                 *,
                 alpha: Number = 1) -> MaskedPair: ...


@overload
def partial_subtract(self: _MaskedPairOrTensor,
                     other: _MaskedPairOrTensor,
                     *,
                     alpha: Number = 1) -> MaskedPair: ...


@overload
def partial_subtract_(self: _MaskedPairOrTensor,
                      other: _MaskedPairOrTensor,
                      *,
                      alpha: Number = 1) -> MaskedPair: ...


def partial_mul(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_mul_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_multiply(self: _MaskedPairOrTensor,
                     other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_multiply_(self: _MaskedPairOrTensor,
                      other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_div(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_div_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_div(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor,
                *,
                rounding_mode: Optional[str] = None) -> MaskedPair: ...


@overload
def partial_div_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor,
                 *,
                 rounding_mode: Optional[str] = None) -> MaskedPair: ...


@overload
def partial_divide(self: _MaskedPairOrTensor,
                   other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_divide_(self: _MaskedPairOrTensor,
                    other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_divide(self: _MaskedPairOrTensor,
                   other: _MaskedPairOrTensor,
                   *,
                   rounding_mode: Optional[str] = None) -> MaskedPair: ...


@overload
def partial_divide_(self: _MaskedPairOrTensor,
                    other: _MaskedPairOrTensor,
                    *,
                    rounding_mode: Optional[str] = None) -> MaskedPair: ...


def partial_floor_divide(self: _MaskedPairOrTensor,
                         other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_floor_divide_(self: _MaskedPairOrTensor,
                          other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_true_divide(self: _MaskedPairOrTensor,
                        other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_true_divide_(self: _MaskedPairOrTensor,
                         other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_logaddexp(self: _MaskedPairOrTensor,
                      other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_logaddexp2(self: _MaskedPairOrTensor,
                       other: _MaskedPairOrTensor) -> MaskedPair: ...


# scaled arithmetics
@overload
def partial_add(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor,
                *,
                alpha: Number = 1,
                scaled: _bool) -> MaskedPair: ...


@overload
def partial_add_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor,
                 *,
                 alpha: Number = 1,
                 scaled: _bool) -> MaskedPair: ...


@overload
def partial_sub(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor,
                *,
                alpha: Number = 1,
                scaled: _bool) -> MaskedPair: ...


@overload
def partial_sub_(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor,
                 *,
                 alpha: Number = 1,
                 scaled: _bool) -> MaskedPair: ...


@overload
def partial_subtract(self: _MaskedPairOrTensor,
                     other: _MaskedPairOrTensor,
                     *,
                     alpha: Number = 1,
                     scaled: _bool) -> MaskedPair: ...


@overload
def partial_subtract_(self: _MaskedPairOrTensor,
                      other: _MaskedPairOrTensor,
                      *,
                      alpha: Number = 1,
                      scaled: _bool) -> MaskedPair: ...


# comparison
def partial_allclose(self: _MaskedPairOrTensor,
                     other: _MaskedPairOrTensor,
                     rtol: _float = 1e-05,
                     atol: _float = 1e-08) -> _bool: ...


def partial_isclose(self: _MaskedPairOrTensor,
                    other: _MaskedPairOrTensor,
                    rtol: _float = 1e-05,
                    atol: _float = 1e-08) -> MaskedPair: ...


def partial_equal(self: _MaskedPairOrTensor,
                  other: _MaskedPairOrTensor) -> _bool: ...


def partial_eq(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_eq_(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_ne(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_ne_(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_not_equal(self: _MaskedPairOrTensor,
                      other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_not_equal_(self: _MaskedPairOrTensor,
                       other: _MaskedPairOrTensor) -> MaskedPair: ...


# min max
def partial_min(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_max(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_minimum(self: _MaskedPairOrTensor,
                    other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_maxium(self: _MaskedPairOrTensor,
                   other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_fmin(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor) -> MaskedPair: ...


def partial_fmax(self: _MaskedPairOrTensor,
                 other: _MaskedPairOrTensor) -> MaskedPair: ...


# ----------------------
# partial binary
# ----------------------
# arithmetics
@overload
def partial_ger(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_mm(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_bmm(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_matmul(self: _MaskedPairOrTensor,
                   other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def linalg_partial_matmul(self: _MaskedPairOrTensor,
                          other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_mv(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_inner(self: _MaskedPairOrTensor,
                  other: _MaskedPairOrTensor) -> MaskedPair: ...


@overload
def partial_outer(self: _MaskedPairOrTensor,
                  other: _MaskedPairOrTensor) -> MaskedPair: ...


# scaled arithmetics
@overload
def partial_ger(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor,
                *,
                scaled: _bool) -> MaskedPair: ...


@overload
def partial_mm(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor,
               *,
               scaled: _bool) -> MaskedPair: ...


@overload
def partial_bmm(self: _MaskedPairOrTensor,
                other: _MaskedPairOrTensor,
                *,
                scaled: _bool) -> MaskedPair: ...


@overload
def partial_matmul(self: _MaskedPairOrTensor,
                   other: _MaskedPairOrTensor,
                   *,
                   scaled: _bool) -> MaskedPair: ...


@overload
def linalg_partial_matmul(self: _MaskedPairOrTensor,
                          other: _MaskedPairOrTensor,
                          *,
                          scaled: _bool) -> MaskedPair: ...


@overload
def partial_mv(self: _MaskedPairOrTensor,
               other: _MaskedPairOrTensor,
               *,
               scaled: _bool) -> MaskedPair: ...


@overload
def partial_inner(self: _MaskedPairOrTensor,
                  other: _MaskedPairOrTensor,
                  *,
                  scaled: _bool) -> MaskedPair: ...


@overload
def partial_outer(self: _MaskedPairOrTensor,
                  other: _MaskedPairOrTensor,
                  *,
                  scaled: _bool) -> MaskedPair: ...
