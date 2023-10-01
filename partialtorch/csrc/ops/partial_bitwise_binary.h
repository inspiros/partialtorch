#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

// ops declaration macros
#define PT_DECLARE_BINARY_OP(NAME, SELF_T, OTHER_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    SELF_T self, OTHER_T other);

#define PT_DECLARE_BINARY_OP_WITH(NAME, SELF_T, OTHER_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(    \
    SELF_T self, OTHER_T other, ARG1);

#define PT_DECLARE_BINARY_OP_WITH2(NAME, SELF_T, OTHER_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(           \
    SELF_T self, OTHER_T other, ARG1, ARG2);

#define PT_DECLARE_BINARY_OP_WITH3(NAME, SELF_T, OTHER_T, ARG1, ARG2, ARG3) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(                 \
    SELF_T self, OTHER_T other, ARG1, ARG2, ARG3);

#define PT_DECLARE_BINARY_OP_(NAME, SELF_T, OTHER_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_( \
    SELF_T self, OTHER_T other);

#define PT_DECLARE_BINARY_OP__WITH(NAME, SELF_T, OTHER_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(  \
    SELF_T self, OTHER_T other, ARG1);

#define PT_DECLARE_BINARY_OP__WITH2(NAME, SELF_T, OTHER_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(         \
    SELF_T self, OTHER_T other, ARG1, ARG2);

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_BINARY_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DECLARE_BINARY_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME)                          \
PT_DECLARE_BINARY_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_BINARY_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1) \
PT_DECLARE_BINARY_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1)                          \
PT_DECLARE_BINARY_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, ARG1)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_BINARY_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2) \
PT_DECLARE_BINARY_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1, ARG2)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2)                          \
PT_DECLARE_BINARY_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1, ARG2)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, ARG1, ARG2)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3) \
PT_DECLARE_BINARY_OP_WITH3(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2, ARG3) \
PT_DECLARE_BINARY_OP_WITH3(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1, ARG2, ARG3)

#define PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_BINARY_OP_(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DECLARE_BINARY_OP_(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)

#define PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME) \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME)                          \
PT_DECLARE_BINARY_OP_(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &)

#define PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME) \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS(NAME)

#define PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_BINARY_OP__WITH(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1) \
PT_DECLARE_BINARY_OP__WITH(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1)

#define PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1)                          \
PT_DECLARE_BINARY_OP__WITH(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1)

#define PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH(NAME, ARG1)

#define PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_BINARY_OP__WITH2(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2) \
PT_DECLARE_BINARY_OP__WITH2(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1, ARG2)

#define PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2)                          \
PT_DECLARE_BINARY_OP__WITH2(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const at::Scalar &, ARG1, ARG2)

#define PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2(NAME, ARG1, ARG2)

#define PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME)                  \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS(NAME)

#define PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME)                  \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(NAME)

#define PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1)                  \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1)

#define PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, ARG1)                  \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(NAME, ARG1)

#define PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2)                  \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2)

#define PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, ARG1, ARG2)                  \
PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(NAME, ARG1, ARG2)

namespace partialtorch {
    namespace ops {
        // logical
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_bitwise_and)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_bitwise_or)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_bitwise_xor)

        // arithmetics
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(
                partial_add, const at::Scalar & alpha = 1, bool scaled = false)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(
                partial_sub, const at::Scalar & alpha = 1, bool scaled = false)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2(
                partial_subtract, const at::Scalar & alpha = 1, bool scaled = false)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_mul)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_multiply)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_div, c10::optional<c10::string_view> rounding_mode = {})
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH(
                partial_divide, c10::optional<c10::string_view> rounding_mode = {})
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_floor_divide)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_true_divide)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_logaddexp)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_logaddexp2)

        // comparison
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_isclose, double rtol = 1e-05, double atol = 1e-08)

        PARTIALTORCH_API bool partial_equal(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                                            const_intrusive_ptr_arg_t<TensorMaskedPair> other);

        PARTIALTORCH_API bool partial_equal(const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                                            const at::Tensor &other);

        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_eq)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_ne)
        PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS(partial_not_equal)

        // min max
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_min)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_max)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_minimum)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_maxium)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_fmin)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(partial_fmax)
    }
}

#undef PT_DECLARE_BINARY_OP
#undef PT_DECLARE_BINARY_OP_WITH
#undef PT_DECLARE_BINARY_OP_WITH2
#undef PT_DECLARE_BINARY_OP_WITH3
#undef PT_DECLARE_BINARY_OP_
#undef PT_DECLARE_BINARY_OP__WITH
#undef PT_DECLARE_BINARY_OP__WITH2
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3
#undef PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS
#undef PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS
#undef PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH
#undef PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH
#undef PT_DECLARE_BINARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_OTHER_SCALAR_OVERLOADS_WITH2
#undef PT_DECLARE_BINARY_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2
#undef PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS
#undef PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH
#undef PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_BINARY_OPS_AND_OPS__FORALL_TENSOR_AND_SCALAR_OVERLOADS_WITH2