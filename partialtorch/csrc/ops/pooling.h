#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

// ~~~~~ ops declaration macros ~~~~~
#define PT_DECLARE_AVG_POOLND_OP(NAME, INPUT_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
        INPUT_T input,                          \
        at::IntArrayRef kernel_size,            \
        at::IntArrayRef stride = 1,             \
        at::IntArrayRef padding = 0,            \
        bool ceil_mode = false,                 \
        bool count_include_pad = true);

#define PT_DECLARE_ADAPTIVE_AVG_POOLND_OP(NAME, INPUT_T, OUTPUT_SIZE_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
        INPUT_T input,                          \
        OUTPUT_SIZE_T output_size);

#define PT_DECLARE_MAX_POOLND_OP(NAME, INPUT_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME( \
        INPUT_T input,                          \
        at::IntArrayRef kernel_size,            \
        at::IntArrayRef stride = 1,             \
        at::IntArrayRef padding = 0,            \
        at::IntArrayRef dilation = 1,           \
        bool ceil_mode = false);

#define PT_DECLARE_MAX_POOLND_WITH_INDICES_OP(NAME, INPUT_T) \
PARTIALTORCH_API std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME( \
        INPUT_T input,                                       \
        at::IntArrayRef kernel_size,                         \
        at::IntArrayRef stride = 1,                          \
        at::IntArrayRef padding = 0,                         \
        at::IntArrayRef dilation = 1,                        \
        bool ceil_mode = false);

#define PT_DECLARE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OP(NAME, INPUT_T) \
PARTIALTORCH_API std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME( \
        INPUT_T input,                                       \
        at::IntArrayRef output_size);

#define PT_DECLARE_FRACTIONAL_MAX_POOLND_OP(NAME, INPUT_T) \
PARTIALTORCH_API std::tuple<c10::intrusive_ptr<TensorMaskedPair>, at::Tensor> NAME( \
        INPUT_T input,                                                       \
        at::IntArrayRef kernel_size,                                         \
        at::IntArrayRef output_size,                                         \
        const at::Tensor &random_samples);

#define PT_DECLARE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_AVG_POOLND_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME, OUTPUT_SIZE_T) \
PT_DECLARE_ADAPTIVE_AVG_POOLND_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T)

#define PT_DECLARE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_MAX_POOLND_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_MAX_POOLND_WITH_INDICES_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

#define PT_DECLARE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_FRACTIONAL_MAX_POOLND_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

namespace partialtorch {
    namespace ops {
        // avg_pool
        PT_DECLARE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(avg_pool1d)

        PT_DECLARE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(avg_pool2d)

        PT_DECLARE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(avg_pool3d)

        PT_DECLARE_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(adaptive_avg_pool1d, at::IntArrayRef)

        PT_DECLARE_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(adaptive_avg_pool2d, at::SymIntArrayRef)

        PT_DECLARE_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS(adaptive_avg_pool3d, at::SymIntArrayRef)

        // max_pool
        PT_DECLARE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(max_pool1d)

        PT_DECLARE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(max_pool2d)

        PT_DECLARE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(max_pool3d)

        PT_DECLARE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(max_pool1d_with_indices)

        PT_DECLARE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(max_pool2d_with_indices)

        PT_DECLARE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(max_pool3d_with_indices)

        PT_DECLARE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(adaptive_max_pool1d)

        PT_DECLARE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(adaptive_max_pool2d)

        PT_DECLARE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS(adaptive_max_pool3d)

        PT_DECLARE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(fractional_max_pool2d)

        PT_DECLARE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS(fractional_max_pool3d)
    }
}

#undef PT_DECLARE_AVG_POOLND_OP
#undef PT_DECLARE_ADAPTIVE_AVG_POOLND_OP
#undef PT_DECLARE_MAX_POOLND_OP
#undef PT_DECLARE_MAX_POOLND_WITH_INDICES_OP
#undef PT_DECLARE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OP
#undef PT_DECLARE_FRACTIONAL_MAX_POOLND_OP
#undef PT_DECLARE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_ADAPTIVE_AVG_POOLND_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_ADAPTIVE_MAX_POOLND_WITH_INDICES_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_FRACTIONAL_MAX_POOLND_OPS_FORALL_TENSOR_OVERLOADS
