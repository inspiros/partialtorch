#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

// ~~~~~ ops declaration macros ~~~~~
#define PT_DECLARE_UPSAMPLE_OP_WITH(NAME, SELF_T, OUTPUT_SIZE_T, ARG1) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(            \
        SELF_T self, OUTPUT_SIZE_T output_size, ARG1);

#define PT_DECLARE_UPSAMPLE_OP_WITH2(NAME, SELF_T, OUTPUT_SIZE_T, ARG1, ARG2) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(                   \
        SELF_T self, OUTPUT_SIZE_T output_size, ARG1, ARG2);

#define PT_DECLARE_UPSAMPLE_OP_WITH3(NAME, SELF_T, OUTPUT_SIZE_T, ARG1, ARG2, ARG3) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(                         \
        SELF_T self, OUTPUT_SIZE_T output_size, ARG1, ARG2, ARG3);

#define PT_DECLARE_UPSAMPLE_OP_WITH4(NAME, SELF_T, OUTPUT_SIZE_T, ARG1, ARG2, ARG3, ARG4) \
PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> NAME(                               \
        SELF_T self, OUTPUT_SIZE_T output_size, ARG1, ARG2, ARG3, ARG4);

#define PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(NAME, OUTPUT_SIZE_T, ARG1) \
PT_DECLARE_UPSAMPLE_OP_WITH(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1)

#define PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(NAME, OUTPUT_SIZE_T, ARG1, ARG2) \
PT_DECLARE_UPSAMPLE_OP_WITH2(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1, ARG2)

#define PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(NAME, OUTPUT_SIZE_T, ARG1, ARG2, ARG3) \
PT_DECLARE_UPSAMPLE_OP_WITH3(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1, ARG2, ARG3)

#define PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH4(NAME, OUTPUT_SIZE_T, ARG1, ARG2, ARG3, ARG4) \
PT_DECLARE_UPSAMPLE_OP_WITH4(NAME, const_intrusive_ptr_arg_t<TensorMaskedPair>, OUTPUT_SIZE_T, ARG1, ARG2, ARG3, ARG4)

namespace partialtorch {
    namespace ops {
        // upsample_nearest
        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                upsample_nearest1d, at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                upsample_nearest1d, c10::SymIntArrayRef,
                c10::optional<double> scales = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                _upsample_nearest_exact1d, at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                _upsample_nearest_exact1d, c10::SymIntArrayRef,
                c10::optional<double> scales = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                upsample_nearest2d, at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_nearest2d, c10::SymIntArrayRef,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                _upsample_nearest_exact2d, at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _upsample_nearest_exact2d, c10::SymIntArrayRef,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                upsample_nearest3d, at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                upsample_nearest3d, c10::SymIntArrayRef,
                c10::optional<double> scales_d = c10::nullopt,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH(
                _upsample_nearest_exact3d, at::OptionalSymIntArrayRef,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _upsample_nearest_exact3d, c10::SymIntArrayRef,
                c10::optional<double> scales_d = c10::nullopt,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        // upsample_lerp
        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_linear1d, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_linear1d, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_bilinear2d, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                upsample_bilinear2d, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _upsample_bilinear2d_aa, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _upsample_bilinear2d_aa, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_trilinear3d, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH4(
                upsample_trilinear3d, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_d = c10::nullopt,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        // partial_upsample_lerp
        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_linear1d, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_linear1d, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_bilinear2d, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                partial_upsample_bilinear2d, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _partial_upsample_bilinear2d_aa, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _partial_upsample_bilinear2d_aa, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_trilinear3d, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH4(
                partial_upsample_trilinear3d, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_d = c10::nullopt,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        // upsample_bicubic
        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                upsample_bicubic2d, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                upsample_bicubic2d, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _upsample_bicubic2d_aa, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _upsample_bicubic2d_aa, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        // partial_upsample_bicubic
        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                partial_upsample_bicubic2d, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                partial_upsample_bicubic2d, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2(
                _partial_upsample_bicubic2d_aa, at::OptionalSymIntArrayRef,
                bool align_corners,
                c10::optional<at::ArrayRef<double>> scale_factors)

        PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3(
                _partial_upsample_bicubic2d_aa, c10::SymIntArrayRef,
                bool align_corners,
                c10::optional<double> scales_h = c10::nullopt,
                c10::optional<double> scales_w = c10::nullopt)
    }
}

#undef PT_DECLARE_UPSAMPLE_OP_WITH
#undef PT_DECLARE_UPSAMPLE_OP_WITH2
#undef PT_DECLARE_UPSAMPLE_OP_WITH3
#undef PT_DECLARE_UPSAMPLE_OP_WITH4
#undef PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH
#undef PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH2
#undef PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH3
#undef PT_DECLARE_UPSAMPLE_OPS_FORALL_TENSOR_OVERLOADS_WITH4
