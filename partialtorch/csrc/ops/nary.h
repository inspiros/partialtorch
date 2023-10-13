#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

// ops declaration macros
#define PT_DECLARE_NARY_OP(NAME, INPUT_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
    INPUT_T tensors);

#define PT_DECLARE_NARY_OP_WITH(NAME, INPUT_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(    \
    INPUT_T tensors, ARG1);

#define PT_DECLARE_NARY_OP_WITH2(NAME, INPUT_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(           \
    INPUT_T tensors, ARG1, ARG2);

#define PT_DECLARE_NARY_OP_WITH3(NAME, INPUT_T, ARG1, ARG2, ARG3) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(                 \
    INPUT_T tensors, ARG1, ARG2, ARG3);

#define PT_DECLARE_NARY_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_NARY_OP(NAME, TensorMaskedPairIntrusivePtrList)

#define PT_DECLARE_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, ARG1) \
PT_DECLARE_NARY_OP_WITH(NAME, TensorMaskedPairIntrusivePtrList, ARG1)

#define PT_DECLARE_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, ARG1, ARG2) \
PT_DECLARE_NARY_OP_WITH2(NAME, TensorMaskedPairIntrusivePtrList, ARG1, ARG2)

#define PT_DECLARE_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, ARG1, ARG2, ARG3) \
PT_DECLARE_NARY_OP_WITH3(NAME, TensorMaskedPairIntrusivePtrList, ARG1, ARG2, ARG3)

namespace partialtorch {
    namespace ops {
        PT_DECLARE_NARY_OPS_FORALL_TENSOR_OVERLOADS(linalg_partial_multi_dot)

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> partial_einsum(
                c10::string_view equation,
                TensorMaskedPairIntrusivePtrList tensors,
                at::OptionalIntArrayRef path = c10::nullopt);
    }
}

#undef PT_DECLARE_NARY_OP
#undef PT_DECLARE_NARY_OP_WITH
#undef PT_DECLARE_NARY_OP_WITH2
#undef PT_DECLARE_NARY_OP_WITH3
#undef PT_DECLARE_NARY_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_NARY_OPS_FORALL_TENSOR_OVERLOADS_WITH3
