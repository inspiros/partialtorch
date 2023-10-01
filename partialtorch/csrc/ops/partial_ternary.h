#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

// ops declaration macros
#define PT_DECLARE_TERNARY_OP(NAME, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME)       \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(                                     \
    SELF_T self, OTHER1_T other1, OTHER2_T other2);

#define PT_DECLARE_TERNARY_OP_WITH(NAME, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(                                          \
    SELF_T self, OTHER1_T other1, OTHER2_T other2, ARG1);

#define PT_DECLARE_TERNARY_OP_WITH2(NAME, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(                                                 \
    SELF_T self, OTHER1_T other1, OTHER2_T other2, ARG1, ARG2);

#define PT_DECLARE_TERNARY_OP_(NAME, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME)       \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                   \
    SELF_T self, OTHER1_T other1, OTHER2_T other2);

#define PT_DECLARE_TERNARY_OP__WITH(NAME, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                        \
    SELF_T self, OTHER1_T other1, OTHER2_T other2, ARG1);

#define PT_DECLARE_TERNARY_OP__WITH2(NAME, SELF_T, OTHER1_T, OTHER2_T, OTHER1_NAME, OTHER2_NAME, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME##_(                                               \
    SELF_T self, OTHER1_T other1, OTHER2_T other2, ARG1, ARG2);

#define PT_DECLARE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, OTHER1_NAME, OTHER2_NAME) \
PT_DECLARE_TERNARY_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME)

#define PT_DECLARE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, OTHER1_NAME, OTHER2_NAME, ARG1) \
PT_DECLARE_TERNARY_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, ARG1)

#define PT_DECLARE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, OTHER1_NAME, OTHER2_NAME, ARG1, ARG2) \
PT_DECLARE_TERNARY_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, ARG1, ARG2)

#define PT_DECLARE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, OTHER1_NAME, OTHER2_NAME) \
PT_DECLARE_TERNARY_OP_(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME)

#define PT_DECLARE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, OTHER1_NAME, OTHER2_NAME, ARG1) \
PT_DECLARE_TERNARY_OP__WITH(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, ARG1)

#define PT_DECLARE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, OTHER1_NAME, OTHER2_NAME, ARG1, ARG2) \
PT_DECLARE_TERNARY_OP__WITH2(NAME, intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, const_intrusive_ptr_arg_t<TensorMaskedPair>, OTHER1_NAME, OTHER2_NAME, ARG1, ARG2)

#define PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS(NAME, OTHER1_NAME, OTHER2_NAME)  \
PT_DECLARE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS(NAME, OTHER1_NAME, OTHER2_NAME)                   \
PT_DECLARE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS(NAME, OTHER1_NAME, OTHER2_NAME)

#define PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, OTHER1_NAME, OTHER2_NAME, ARG1)  \
PT_DECLARE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, OTHER1_NAME, OTHER2_NAME, ARG1)                   \
PT_DECLARE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH(NAME, OTHER1_NAME, OTHER2_NAME, ARG1)

#define PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, OTHER1_NAME, OTHER2_NAME, ARG1, ARG2)  \
PT_DECLARE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, OTHER1_NAME, OTHER2_NAME, ARG1, ARG2)                   \
PT_DECLARE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2(NAME, OTHER1_NAME, OTHER2_NAME, ARG1, ARG2)

namespace partialtorch {
    namespace ops {
        PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_addmm, mat1, mat2,
                const at::Scalar &beta = 1, const at::Scalar &alpha = 1)

        PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_addbmm, batch1, batch2,
                const at::Scalar &beta = 1, const at::Scalar &alpha = 1)

        PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_baddbmm, batch1, batch2,
                const at::Scalar &beta = 1, const at::Scalar &alpha = 1)

        PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_addmv, mat, vec,
                const at::Scalar &beta = 1, const at::Scalar &alpha = 1)

        PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2(
                partial_addr, vec1, vec2,
                const at::Scalar &beta = 1, const at::Scalar &alpha = 1)

        PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                partial_addcmul, tensor1, tensor2,
                const at::Scalar &alpha = 1)

        PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH(
                partial_addcdiv, tensor1, tensor2,
                const at::Scalar &alpha = 1)
    }
}

#undef PT_DECLARE_TERNARY_OP
#undef PT_DECLARE_TERNARY_OP_WITH
#undef PT_DECLARE_TERNARY_OP_WITH2
#undef PT_DECLARE_TERNARY_OP_
#undef PT_DECLARE_TERNARY_OP__WITH
#undef PT_DECLARE_TERNARY_OP__WITH2
#undef PT_DECLARE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_TERNARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_TERNARY_OPS__FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_TERNARY_OPS_AND_OPS__FORALL_TENSOR_OVERLOADS_WITH2
