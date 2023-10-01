#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

// ~~~~~ ops declaration macros ~~~~~
#define PT_DECLARE_SCALED_CONVND_OP(NAME, INPUT_T, PADDING_T, PADDING_DEFAULT) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(                    \
        INPUT_T input,                                                         \
        const at::Tensor &weight,                                              \
        const c10::optional<at::Tensor> &bias = {},                            \
        at::IntArrayRef stride = 1,                                            \
        PADDING_T padding = PADDING_DEFAULT,                                   \
        at::IntArrayRef dilation = 1,                                          \
        int64_t groups = 1,                                                    \
        bool scaled = false);

#define PT_DECLARE_SCALED_CONV_TRANSPOSEND_OP(NAME, INPUT_T) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(  \
        INPUT_T input,                                       \
        const at::Tensor &weight,                            \
        const c10::optional<at::Tensor> &bias = {},          \
        at::IntArrayRef stride = 1,                          \
        at::IntArrayRef padding = 0,                         \
        at::IntArrayRef output_padding = 0,                  \
        int64_t groups = 1,                                  \
        at::IntArrayRef dilation = 1,                        \
        bool scaled = false);

#define PT_DECLARE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(NAME, PADDING_T, PADDING_DEFAULT) \
PT_DECLARE_SCALED_CONVND_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, PADDING_T, PADDING_DEFAULT)

#define PT_DECLARE_SCALED_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(NAME) \
PT_DECLARE_SCALED_CONV_TRANSPOSEND_OP(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>)

namespace partialtorch {
    namespace ops {
        // convolution
        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const_intrusive_ptr_arg_t < TensorMaskedPair > input,
        const at::Tensor &weight,
        const c10::optional<at::Tensor> &bias,
                at::IntArrayRef
        stride,
        at::IntArrayRef padding,
                at::IntArrayRef
        dilation,
        bool transposed,
                at::IntArrayRef
        output_padding,
        int64_t groups,
        bool scaled = false
        );

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> partial_convolution(
                const_intrusive_ptr_arg_t < TensorMaskedPair > input,
        const at::Tensor &weight,
        const c10::optional<at::Tensor> &bias,
                at::IntArrayRef
        stride,
        at::SymIntArrayRef padding,
                at::IntArrayRef
        dilation,
        bool transposed,
                at::SymIntArrayRef
        output_padding,
        int64_t groups,
        bool scaled = false
        );

        // convnd
        PT_DECLARE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(partial_conv1d, at::IntArrayRef, 0)

        PT_DECLARE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(partial_conv1d, c10::string_view, "valid")

        PT_DECLARE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(partial_conv2d, at::IntArrayRef, 0)

        PT_DECLARE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(partial_conv2d, c10::string_view, "valid")

        PT_DECLARE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(partial_conv3d, at::IntArrayRef, 0)

        PT_DECLARE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS(partial_conv3d, c10::string_view, "valid")

        // conv_transposend
        PT_DECLARE_SCALED_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(partial_conv_transpose1d)

        PT_DECLARE_SCALED_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(partial_conv_transpose2d)

        PT_DECLARE_SCALED_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS(partial_conv_transpose3d)
    }
}

#undef PT_DECLARE_CONVND_OP
#undef PT_DECLARE_SCALED_CONVND_OP
#undef PT_DECLARE_CONV_TRANSPOSEND_OP
#undef PT_DECLARE_SCALED_CONV_TRANSPOSEND_OP
#undef PT_DECLARE_SCALED_CONVND_OPS_FORALL_TENSOR_OVERLOADS
#undef PT_DECLARE_SCALED_CONV_TRANSPOSEND_OPS_FORALL_TENSOR_OVERLOADS
