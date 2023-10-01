#pragma once

#include <ATen/ATen.h>

#include "../MaskedPair.h"
#include "../macros.h"
#include "utils/torch_library_utils.h"

namespace partialtorch {
    namespace ops {
        PARTIALTORCH_API void _backward(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                TensorMaskedPairIntrusivePtrList inputs,
                const c10::optional<at::Tensor> &gradient = c10::nullopt,
                c10::optional<bool> retain_graph = c10::nullopt,
                bool create_graph = false);

        PARTIALTORCH_API void _backward(
                const_intrusive_ptr_arg_t<TensorMaskedPair> self,
                at::TensorList inputs,
                const c10::optional<at::Tensor> &gradient = c10::nullopt,
                c10::optional<bool> retain_graph = c10::nullopt,
                bool create_graph = false);

        PARTIALTORCH_API void _backward(
                const at::Tensor &self,
                TensorMaskedPairIntrusivePtrList inputs,
                const c10::optional<at::Tensor> &gradient = c10::nullopt,
                c10::optional<bool> retain_graph = c10::nullopt,
                bool create_graph = false);
    }
}
