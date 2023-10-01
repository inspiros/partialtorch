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

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_BINARY_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>) \
PT_DECLARE_BINARY_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_BINARY_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1) \
PT_DECLARE_BINARY_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_BINARY_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2) \
PT_DECLARE_BINARY_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1, ARG2)

#define PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3) \
PT_DECLARE_BINARY_OP_WITH3(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, ARG1, ARG2, ARG3) \
PT_DECLARE_BINARY_OP_WITH3(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const at::Tensor &, ARG1, ARG2, ARG3)

namespace partialtorch {
    namespace ops {
        // arithmetics
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                partial_ger, bool scaled = false)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                partial_mm, bool scaled = false)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                partial_bmm, bool scaled = false)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                partial_matmul, bool scaled = false)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                partial_mv, bool scaled = false)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                partial_inner, bool scaled = false)
        PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                partial_outer, bool scaled = false)
    }
}

#undef PT_DECLARE_BINARY_OP
#undef PT_DECLARE_BINARY_OP_WITH
#undef PT_DECLARE_BINARY_OP_WITH2
#undef PT_DECLARE_BINARY_OP_WITH3
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_BINARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3
