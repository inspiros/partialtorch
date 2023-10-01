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
                double eps);
    }
}
