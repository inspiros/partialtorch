#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

namespace partialtorch {
    namespace ops {
        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> batch_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const c10::optional<at::Tensor> &weight,
                const c10::optional<at::Tensor> &bias,
                const c10::optional<at::Tensor> &running_mean,
                const c10::optional<at::Tensor> &running_var,
                bool training,
                double momentum,
                double eps,
                bool cudnn_enabled);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> instance_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const c10::optional<at::Tensor> &weight,
                const c10::optional<at::Tensor> &bias,
                const c10::optional<at::Tensor> &running_mean,
                const c10::optional<at::Tensor> &running_var,
                bool use_input_stats,
                double momentum,
                double eps,
                bool cudnn_enabled);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> layer_norm(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                at::SymIntArrayRef normalized_shape,
                const c10::optional<at::Tensor> &weight = c10::nullopt,
                const c10::optional<at::Tensor> &bias = c10::nullopt,
                double eps = 1e-5,
                bool cudnn_enabled = true);
    }
}
