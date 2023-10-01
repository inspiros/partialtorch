#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

namespace partialtorch {
    namespace ops {
        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> partial_linear(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias = {},
                bool scaled = false);

        PARTIALTORCH_API c10::intrusive_ptr<TensorMaskedPair> partial_bilinear(
                const_intrusive_ptr_arg_t<TensorMaskedPair> input1,
                const_intrusive_ptr_arg_t<TensorMaskedPair> input2,
                const at::Tensor &weight,
                const c10::optional<at::Tensor> &bias = {},
                bool scaled = false);
    }
}
